#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot (R1/S1) breakout with 4h volume confirmation and 1d EMA50 trend filter.
# Long when price breaks above R1 with volume > 1.5x 20-period average and price > 1d EMA50.
# Short when price breaks below S1 with volume > 1.5x 20-period average and price < 1d EMA50.
# Exit when price crosses back below R1 (for long) or above S1 (for short).
# Uses Camarilla pivot for intraday support/resistance, volume to confirm breakout strength,
# and trend filter to avoid counter-trend trades. Target: 60-150 total trades over 4 years (15-37/year) for 1h.

name = "1h_Camarilla_R1S1_4hVolume_1dEMA50"
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
    
    # 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points (using previous day's OHLC)
    # For intraday, we use previous day's data; for first bar of day, use previous available day
    # We'll approximate by using rolling window of 24 hours (96 bars of 15m, but we're on 1h)
    # Instead, we use previous day's high, low, close from 1d data aligned to 1h
    # However, for simplicity and to avoid look-ahead, we calculate pivots from 1d OHLC of previous day
    # We'll shift 1d data by 1 to ensure we only use previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan  # First day has no previous
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    rng = prev_high_1d - prev_low_1d
    camarilla_r1 = prev_close_1d + rng * 1.1 / 12
    camarilla_s1 = prev_close_1d - rng * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_4h = df_4h['volume'].values
    vol_ma20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20_4h)
    volume_filter = volume > (1.5 * vol_ma20_4h_aligned)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1, volume spike, above 1d EMA50
            long_cond = (close[i] > camarilla_r1_aligned[i]) and volume_filter[i] and (close[i] > ema50_1d_aligned[i])
            # Short conditions: price breaks below S1, volume spike, below 1d EMA50
            short_cond = (close[i] < camarilla_s1_aligned[i]) and volume_filter[i] and (close[i] < ema50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses back below R1
            if close[i] < camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses back above S1
            if close[i] > camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals