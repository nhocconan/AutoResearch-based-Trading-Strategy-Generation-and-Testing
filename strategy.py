#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_volume_trend_v1
# Strategy: Daily Camarilla pivot breakout with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Uses weekly EMA200 to determine long-term trend and daily Camarilla pivot levels (R4/S4 for breakouts) filtered by volume spikes. 
# This strategy targets multi-week momentum bursts while filtering counter-trend noise and avoiding overtrading via daily timeframe.
# Designed to work in both bull and bear markets by using trend filter to align with higher timeframe momentum.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA200 for long-term trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Load daily data for OHLC calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align daily data to daily timeframe (no shift needed as we use previous day's levels)
    # But we still use align_htf_to_ltf to ensure proper handling of data gaps
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA200
        uptrend_1w = price_close > ema_200_1w_aligned[i]
        downtrend_1w = price_close < ema_200_1w_aligned[i]
        
        # Camarilla breakout signals (using previous day's levels)
        breakout_up = price_close > r4_1d_aligned[i]   # Break above R4
        breakdown_down = price_close < s4_1d_aligned[i]  # Break below S4
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Break above R4 with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1w
        
        # Short: Break below S4 with volume in downtrend
        short_signal = breakdown_down and vol_confirmed and downtrend_1w
        
        # Exit when price returns to the daily pivot level
        exit_long = position == 1 and price_close < pivot_1d_aligned[i]
        exit_short = position == -1 and price_close > pivot_1d_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals