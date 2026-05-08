#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = (close_1w > ema34_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get 1d data once for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation (R3/S3 levels)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla pivot levels calculation (R3 and S3)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 2)  # R3 level
    s3 = pivot - (range_val * 1.1 / 2)  # S3 level
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection: current volume > 2.0 * 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma30 * 2.0)
    
    # Price distance filter: require breakout to be at least 0.3% above/below level
    price_above_r3 = close > r3_12h * 1.003
    price_below_s3 = close < s3_12h * 0.997
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and 1w uptrend
            long_cond = (price_above_r3[i] and vol_spike[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and 1w downtrend
            short_cond = (price_below_s3[i] and vol_spike[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses back below R3 (mean reversion)
            if close[i] < r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above S3 (mean reversion)
            if close[i] > s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout on 12h timeframe with 1-week EMA34 trend filter and volume confirmation.
# Uses tighter volume threshold (2.0x 30-period MA) and price distance filter (0.3%) to reduce false breakouts.
# Position size 0.25 to manage risk. Designed for 12-37 trades/year to avoid fee drag.
# Targets both bull and bear markets via trend filter and mean-reversion exits.