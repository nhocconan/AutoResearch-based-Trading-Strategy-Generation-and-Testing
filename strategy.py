#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla R3 and S3 levels (more reliable than R1/S1)
    range_val = prev_high - prev_low
    r3 = prev_close + (range_val * 1.1 / 4)
    s3 = prev_close - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = (close_1d > ema34_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    # Price distance filter: require breakout to be at least 0.4% above/below level
    price_above_r3 = close > r3_4h * 1.004
    price_below_s3 = close < s3_4h * 0.996
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and 1d uptrend
            long_cond = (price_above_r3[i] and vol_spike[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and 1d downtrend
            short_cond = (price_below_s3[i] and vol_spike[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.28
                position = 1
            elif short_cond:
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Long exit: price reverses back below R3 (mean reversion)
            if close[i] < r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Short exit: price reverses back above S3 (mean reversion)
            if close[i] > s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with volume confirmation and 1d EMA34 trend filter on 4h timeframe.
# Uses Camarilla R3/S3 levels (more robust than R1/S1) for stronger breakout signals.
# Volume spike >2x 20-period average ensures institutional participation.
# Price distance filter (0.4%) avoids false breakouts from noise.
# Trend filter ensures alignment with daily bias.
# Target: 25-45 trades/year to minimize fee decay while capturing meaningful moves.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at S3/R3).