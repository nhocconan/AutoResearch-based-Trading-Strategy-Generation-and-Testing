#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = (close_1w > ema34_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get daily data for Camarilla pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily Camarilla pivot levels (R3, S3) based on previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and levels for previous day
    pivot = (high_1d + low_1d + close_1d) / 3
    r3 = pivot + 1.1 * (high_1d - low_1d)
    s3 = pivot - 1.1 * (high_1d - low_1d)
    
    # Shift to get previous day's levels (available at close of previous day)
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan
    
    # Align to lower timeframe
    r3_prev_aligned = align_htf_to_ltf(prices, df_1d, r3_prev)
    s3_prev_aligned = align_htf_to_ltf(prices, df_1d, s3_prev)
    
    # Daily volume spike detection: current volume > 2.0 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20d)
    vol_spike = volume > (vol_ma20d_aligned * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_prev_aligned[i]) or np.isnan(s3_prev_aligned[i]) or 
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma20d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and weekly uptrend
            long_cond = (close[i] > r3_prev_aligned[i] and vol_spike[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and weekly downtrend
            short_cond = (close[i] < s3_prev_aligned[i] and vol_spike[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 (mean reversion to opposite level)
            if close[i] < s3_prev_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 (mean reversion to opposite level)
            if close[i] > r3_prev_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout on daily timeframe with volume confirmation and weekly trend filter.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at opposite level).
# Weekly EMA34 ensures alignment with longer-term trend, reducing counter-trend trades.
# Volume spike filter (2.0x 20-day average) ensures momentum confirmation.
# Target: 10-25 trades/year to minimize fee decay while capturing significant moves.
# Uses 1d timeframe for execution with weekly trend filter to avoid overtrading.