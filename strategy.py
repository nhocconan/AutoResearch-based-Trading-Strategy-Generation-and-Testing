#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3S3_Breakout_4hTrend_VolumeS"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_4h = (close_4h > ema20_4h).astype(float)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
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
    
    # Align Camarilla levels to 1h timeframe
    r3_1h = align_htf_to_ltf(prices, df_1d, r3)
    s3_1h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Price distance filter: require breakout to be at least 0.3% above/below level
    price_above_r3 = close > r3_1h * 1.003
    price_below_s3 = close < s3_1h * 0.997
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or np.isnan(trend_4h_aligned[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and 4h uptrend during session
            long_cond = (price_above_r3[i] and vol_spike[i] and trend_4h_aligned[i] > 0.5 and session_filter[i])
            
            # Short entry: price breaks below S3 with volume spike and 4h downtrend during session
            short_cond = (price_below_s3[i] and vol_spike[i] and trend_4h_aligned[i] < 0.5 and session_filter[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reverses back below R3 (mean reversion)
            if close[i] < r3_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverses back above S3 (mean reversion)
            if close[i] > s3_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout on 1h timeframe with 4h EMA20 trend filter and volume confirmation.
# Uses session filter (08-20 UTC) to reduce noise trades and improve win rate.
# Position size fixed at 0.20 to manage risk. Target: 15-30 trades/year to avoid fee drag.
# Designed to work in both bull and bear markets by combining mean-reversion exits with trend-filtered entries.