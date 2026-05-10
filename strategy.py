# NOTE: This strategy is provided for educational purposes only. Past performance does not guarantee future results.
# Users are responsible for complying with local laws and regulations.
#!/usr/bin/env python3
"""
1h_RSI_4H_Trend_Filter
Hypothesis: In strong 4h trends (EMA50), 1h RSI pullbacks offer high-probability entries with favorable risk-reward.
Works in bull markets via buying dips in uptrends and in bear markets via selling rallies in downtrends.
Uses 4h EMA50 for trend and 1h RSI(14) for entry timing, with volume confirmation to avoid false signals.
Designed for low trade frequency (target: 60-150 trades over 4 years) to minimize fee drag.
"""

name = "1h_RSI_4H_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14) and 4h EMA50 (50)
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h trend filter
        uptrend_4h = close[i] > ema50_4h_aligned[i]
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        
        # Volume filter: current volume > 1.5x 4h average volume (scaled to 1h)
        # 4h = 4 x 1h bars, so scale 4h volume to 1h equivalent
        vol_4h_avg = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
        vol_4h_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_4h_avg)
        vol_1h_equiv = vol_4h_avg_aligned[i] / 4.0
        volume_filter = volume[i] > vol_1h_equiv * 1.5
        
        if position == 0:
            # Long entry: RSI < 40 (pullback) + uptrend + volume
            if rsi[i] < 40 and uptrend_4h and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short entry: RSI > 60 (pullback) + downtrend + volume
            elif rsi[i] > 60 and downtrend_4h and volume_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI > 60 or trend fails
            if rsi[i] > 60 or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 40 or trend fails
            if rsi[i] < 40 or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals