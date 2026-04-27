# Strategy hypothesis: 12h timeframe with Camarilla pivot breakout (R3/S3) confirmed by 1d trend (EMA34) and volume spike. Designed to capture strong trends while avoiding chop. Uses discrete position sizing to minimize fee churn. Target: 15-25 trades/year.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Previous 12h period's high, low, close (for completed period)
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Calculate Camarilla Pivot Levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 1.1000 / 4
    s3 = prev_close - range_hl * 1.1000 / 4
    r4 = prev_close + range_hl * 1.1000 / 2
    s4 = prev_close - range_hl * 1.1000 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_12h, r3)
    s3_12h = align_htf_to_ltf(prices, df_12h, s3)
    r4_12h = align_htf_to_ltf(prices, df_12h, r4)
    s4_12h = align_htf_to_ltf(prices, df_12h, s4)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(r4_12h[i]) or 
            np.isnan(s4_12h[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 + 1d uptrend + volume spike
            if (close[i] > r3_12h[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + 1d downtrend + volume spike
            elif (close[i] < s3_12h[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S3 (reversal) or trend changes
            if (close[i] < s3_12h[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 (reversal) or trend changes
            if (close[i] > r3_12h[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0