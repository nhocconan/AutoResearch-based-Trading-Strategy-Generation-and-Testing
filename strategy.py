#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-day Camarilla pivot direction and volume confirmation
# Long when price breaks above Donchian high and is above daily Camarilla pivot (R1)
# Short when price breaks below Donchian low and is below daily Camarilla pivot (S1)
# Uses volume surge (>1.5x daily average) to confirm breakout strength
# Designed for 6h timeframe to target 50-150 trades over 4 years (12-37/year)
# Works in bull/bear: Camarilla pivot adapts to volatility, volume filter avoids fakeouts

name = "6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20) on 6h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Daily Camarilla Pivot Levels (from prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical Price and range for prior day
    tp_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels for current day (based on prior day)
    # R4 = close + 1.5 * range * 1.1/2
    # R3 = close + 1.25 * range * 1.1/2
    # R2 = close + 1.0 * range * 1.1/2
    # R1 = close + 0.5 * range * 1.1/2
    # PP = (high + low + close) / 3
    # S1 = close - 0.5 * range * 1.1/2
    # S2 = close - 1.0 * range * 1.1/2
    # S3 = close - 1.25 * range * 1.1/2
    # S4 = close - 1.5 * range * 1.1/2
    
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    camarilla_pp = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):  # Start from 1 to use prior day
        camarilla_pp[i] = tp_1d[i-1]
        camarilla_r1[i] = tp_1d[i-1] + 0.5 * range_1d[i-1] * 1.1 / 2
        camarilla_s1[i] = tp_1d[i-1] - 0.5 * range_1d[i-1] * 1.1 / 2
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(10, len(vol_1d)):  # 10-day average
        vol_ma_1d[i] = np.mean(vol_1d[i-10:i])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 10)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Check exits and stoploss (2.5 * ATR-like using daily range)
        if position == 1:  # long position
            # Exit: breakdown below Donchian low or stoploss
            if (close[i] < donchian_low[i] or 
                close[i] < entry_price - 2.5 * (camarilla_r1_aligned[i] - camarilla_s1_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: breakout above Donchian high or stoploss
            if (close[i] > donchian_high[i] or 
                close[i] > entry_price + 2.5 * (camarilla_r1_aligned[i] - camarilla_s1_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: breakout above Donchian high and above Camarilla R1
                if close[i] > donchian_high[i] and close[i] > camarilla_r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low and below Camarilla S1
                elif close[i] < donchian_low[i] and close[i] < camarilla_s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals