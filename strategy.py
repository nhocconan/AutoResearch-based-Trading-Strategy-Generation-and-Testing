#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Camarilla_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Previous day's high, low, close
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first value to nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_hl * 1.1 / 12)
    R2 = pivot + (range_hl * 1.1 / 6)
    R3 = pivot + (range_hl * 1.1 / 4)
    R4 = pivot + (range_hl * 1.1 / 2)
    S1 = pivot - (range_hl * 1.1 / 12)
    S2 = pivot - (range_hl * 1.1 / 6)
    S3 = pivot - (range_hl * 1.1 / 4)
    S4 = pivot - (range_hl * 1.1 / 2)
    
    # Align levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    R2_6h = align_htf_to_ltf(prices, df_1d, R2)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    S2_6h = align_htf_to_ltf(prices, df_1d, S2)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA for trend
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(pivot_6h[i]) or np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or \
           np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        # Weekly trend filter
        weekly_uptrend = price > ema_20_1w_aligned[i]
        weekly_downtrend = price < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume in weekly uptrend
            if price > R4_6h[i] and volume_ok and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with volume in weekly downtrend
            elif price < S4_6h[i] and volume_ok and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to pivot or weekly trend changes
            if price <= pivot_6h[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to pivot or weekly trend changes
            if price >= pivot_6h[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals