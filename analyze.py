# analyze_drowsy_csv.py
import argparse, os, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    if w<=1: return x.values.astype(float)
    return np.convolve(x.values.astype(float), np.ones(w)/w, mode='same')

def make_segments(df):
    """연속된 동일 label 구간을 세그먼트로 요약"""
    segs=[]
    if df.empty: return pd.DataFrame(columns=["seg_id","label","start_frame","end_frame","n_frames","avg_prob"])
    cur_label = df.loc[0,"label"]
    start = int(df.loc[0,"frame_idx"])
    probs=[float(df.loc[0,"prob"])]
    for i in range(1,len(df)):
        if df.loc[i,"label"]==cur_label and int(df.loc[i,"frame_idx"])==int(df.loc[i-1,"frame_idx"])+1:
            probs.append(float(df.loc[i,"prob"]))
        else:
            end=int(df.loc[i-1,"frame_idx"])
            segs.append((cur_label,start,end,end-start+1,float(np.mean(probs))))
            cur_label=df.loc[i,"label"]
            start=int(df.loc[i,"frame_idx"])
            probs=[float(df.loc[i,"prob"])]
    # last
    end=int(df.loc[len(df)-1,"frame_idx"])
    segs.append((cur_label,start,end,end-start+1,float(np.mean(probs))))
    out=pd.DataFrame(segs,columns=["label","start_frame","end_frame","n_frames","avg_prob"])
    out.insert(0,"seg_id",range(1,len(out)+1))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("csv_path", help="예: input.csv (infer_video_tflite.py 결과)")
    ap.add_argument("--fps", type=float, default=None, help="실제 영상 FPS (모르면 생략: 평균 latency로 추정)")
    ap.add_argument("--smooth", type=int, default=9, help="이동평균(프레임) 윈도우(홀수 추천). 1이면 무시")
    ap.add_argument("--drowsy_label", default="DROWSY", help="졸림 레이블 이름(기본: DROWSY)")
    ap.add_argument("--plot", action="store_true", help="타임시리즈 그래프 PNG 저장")
    args=ap.parse_args()

    p=pathlib.Path(args.csv_path)
    df=pd.read_csv(p)

    # 기본 정렬/정합
    df=df.sort_values("frame_idx").reset_index(drop=True)

    # 속도/지연 요약
    est_fps = 1000.0/df["latency_ms"].mean() if not df["latency_ms"].isna().all() else np.nan
    fps = args.fps if args.fps else est_fps
    duration_s = (df["frame_idx"].iloc[-1]-df["frame_idx"].iloc[0]+1)/fps if fps and fps>0 else len(df)/est_fps

    summary = {
        "frames": len(df),
        "avg_latency_ms": df["latency_ms"].mean(),
        "median_latency_ms": df["latency_ms"].median(),
        "fps_used": fps,
        "duration_sec_est": duration_s
    }

    # 레이블 분포
    label_counts = df["label"].value_counts().rename_axis("label").reset_index(name="frames")
    label_counts["ratio_%"] = (label_counts["frames"]/len(df)*100).round(2)

    # 연속 세그먼트
    segs = make_segments(df)
    if fps and fps>0:
        segs["duration_s"] = segs["n_frames"]/fps
    else:
        segs["duration_s"] = np.nan

    # DROWSY 점수(바이너리 가정) 만들기: 예) 예측이 DROWSY면 prob, 아니면 1-prob
    dlab = args.drowsy_label
    dscore = np.where(df["label"]==dlab, df["prob"].astype(float), 1.0-df["prob"].astype(float))
    df["drowsy_score_raw"]=dscore
    df["drowsy_score_smooth"]=moving_average(df["drowsy_score_raw"], args.smooth)

    # 단순 임계치 이벤트 탐지 (스무딩 점수가 thr 이상으로 k 프레임 이상 지속)
    thr=0.7; k=max(5, args.smooth//2)  # 기본값 예시
    above = df["drowsy_score_smooth"]>=thr
    events=[]
    i=0
    while i<len(df):
        if above.iloc[i]:
            j=i
            while j+1<len(df) and above.iloc[j+1]:
                j+=1
            length=j-i+1
            if length>=k:
                events.append((int(df.loc[i,"frame_idx"]), int(df.loc[j,"frame_idx"]), length,
                               float(df["drowsy_score_smooth"].iloc[i:j+1].mean())))
            i=j+1
        else:
            i+=1
    events_df=pd.DataFrame(events, columns=["start_frame","end_frame","n_frames","avg_drowsy_score"])
    if fps and fps>0 and len(events_df):
        events_df["duration_s"]=events_df["n_frames"]/fps

    # 출력 저장
    stem = p.stem
    outdir = p.parent / "analyze" / stem
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / f"{stem}_summary.json"
    label_path  = outdir / f"{stem}_label_counts.csv"
    segs_path   = outdir / f"{stem}_segments.csv"
    events_path = outdir / f"{stem}_drowsy_events.csv"

    import json
    with open(summary_path,"w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    label_counts.to_csv(label_path, index=False, encoding="utf-8")
    segs.to_csv(segs_path, index=False, encoding="utf-8")
    events_df.to_csv(events_path, index=False, encoding="utf-8")

    print("=== SUMMARY ===")
    for k,v in summary.items():
        print(f"{k:>18}: {v}")
    print("\nLabel counts:")
    print(label_counts)
    print(f"\nSaved: {summary_path}\n       {label_path}\n       {segs_path}\n       {events_path}")

    # (옵션) 그래프
    if args.plot:
        png1 = outdir / f"{stem}_drowsy_score.png"
        plt.figure()
        plt.plot(df["frame_idx"], df["drowsy_score_raw"], label="raw")
        plt.plot(df["frame_idx"], df["drowsy_score_smooth"], label=f"smooth({args.smooth})")
        plt.xlabel("frame_idx"); plt.ylabel("drowsy_score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(png1, dpi=150)
        print(f"Saved plot: {png1}")

if __name__=="__main__":
    main()
