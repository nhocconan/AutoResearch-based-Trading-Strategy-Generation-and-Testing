#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter and Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = (close_1w > ema34_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Previous week's OHLC for Camarilla calculation (R3/S3 levels)
    prev_high = np.roll(df_1w['high'].values, 1)
    prev_low = np.roll(df_1w['low'].values, 1)
    prev_close = np.roll(df_1w['close'].values, 1)
    prev_high[0] = df_1w['high'].values[0]
    prev_low[0] = df_1w['low'].values[0]
    prev_close[0] = df_1w['close'].values[0]
    
    # Camarilla pivot levels calculation (R3 and S3)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 2)  # R3 level
    s3 = pivot - (range_val * 1.1 / 2)  # S3 level
    
    # Align Camarilla levels to 1d timeframe
    r3_1d = align_htf_to_ltf(prices, df_1w, r3)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume spike detection: current volume > 2.5 * 30-period average (more selective)
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma30 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and 1w uptrend
            long_cond = (close[i] > r3_1d[i] and vol_spike[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and 1w downtrend
            short_cond = (close[i] < s3_1d[i] and vol_spike[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal signal)
            if close[i] < s3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above R3 (reversal signal)
            if close[i] > r3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout strategy with volume spike confirmation and 1w EMA34 trend filter on 1d timeframe.
# Uses weekly timeframe for trend filter and pivot levels to capture major market structure.
# Enters long when price breaks above weekly R3 with volume spike and 1w uptrend (close > EMA34).
# Enters short when price breaks below weekly S3 with volume spike and 1w downtrend (close < EMA34).
# Exits when price reverses back through S3/R3 respectively.
# Uses 30-period volume MA with 2.5x threshold for stricter volume confirmation.
# Targets 10-25 trades/year on 1d timeframe to avoid overtrading. Works in bull markets (trend-following breakouts)
# and bear markets (reversal breakouts from extreme levels). Uses discrete sizing (0.25) to minimize churn.