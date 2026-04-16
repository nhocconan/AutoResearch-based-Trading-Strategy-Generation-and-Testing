#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot R1/S1 breakout with volume confirmation and 1d choppiness regime filter.
# Long when price > 1d Camarilla R1, 4h volume > 1.5x median, and 1d CHOP > 61.8 (range market).
# Short when price < 1d Camarilla S1, same volume condition, and 1d CHOP > 61.8.
# Exit when price crosses the 1d Camarilla pivot point (middle).
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear: choppiness filter ensures we only trade in ranging markets where mean reversion at pivot levels is effective.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d data once before loop for Camarilla pivot and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla pivot levels (R1, S1, pivot) and choppiness ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # R1 = C + ((H - L) * 1.1 / 12)
    # S1 = C - ((H - L) * 1.1 / 12)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12.0)
    s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12.0)
    
    # Calculate 1d choppiness index (CHOP) - range: 0-100, >61.8 = ranging, <38.2 = trending
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index"""
        atr = np.zeros_like(close_arr)
        tr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = np.abs(high_arr[i] - close_arr[i-1])
            lc = np.abs(low_arr[i] - close_arr[i-1])
            tr[i] = max(hl, hc, lc)
        # True range for first bar
        tr[0] = high_arr[0] - low_arr[0]
        # ATR calculation
        atr[period-1] = np.mean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Chop calculation
        chop = np.full_like(close_arr, np.nan, dtype=np.float64)
        for i in range(period-1, len(close_arr)):
            atr_sum = np.sum(atr[i-period+1:i+1])
            hh = np.max(high_arr[i-period+1:i+1])
            ll = np.min(low_arr[i-period+1:i+1])
            if hh != ll and atr_sum > 0:
                chop[i] = 100 * np.log10(atr_sum / np.log10(period) / (hh - ll))
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_filter = chop_1d > 61.8  # Only trade in ranging markets
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Volume median for spike detection ===
    vol_4h = df_4h['volume'].values
    vol_median_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (4h)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20)  # 1d lookback, 4h volume median(20)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(chop_filter_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        chop_ok = chop_filter_aligned[i] > 0.5  # Boolean as float
        vol_median = vol_median_aligned[i]
        vol_4h = vol_4h_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below pivot (mean reversion to middle)
            if price < pivot:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above pivot (mean reversion to middle)
            if price > pivot:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 4h volume > 1.5x median volume
            volume_spike = vol_4h > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above Camarilla R1 AND volume spike AND chop filter (ranging market)
            if price > r1 and volume_spike and chop_ok:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S1 AND volume spike AND chop filter (ranging market)
            elif price < s1 and volume_spike and chop_ok:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Camarilla_R1S1_4hVolumeSpike1.5x_1dChop61.8_v1"
timeframe = "4h"
leverage = 1.0