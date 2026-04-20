#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_R1S1_Breakout_Volume_ADXFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1w Camarilla Pivot Points (previous week) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Classic pivot (same for Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels - Focus on R1/S1 for breakouts
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Volume Confirmation (12h) ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === ADX Filter (12h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * plus_dm_ma / np.where(tr_ma > 0, tr_ma, np.nan)
    minus_di = 100 * minus_dm_ma / np.where(tr_ma > 0, tr_ma, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), np.nan)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout at R1/S1 with volume confirmation and ADX > 25 (trending)
            if close_val > r1_val and vol_ratio_val > 2.5 and adx_val > 25:
                # Break above R1
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif close_val < s1_val and vol_ratio_val > 2.5 and adx_val > 25:
                # Break below S1
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: stop loss or return to S1
            if close_val <= entry_price - 1.5 * (prices['high'].iloc[i] - prices['low'].iloc[i]):  # Tighter stop
                # Stop loss hit
                signals[i] = 0.0
                position = 0
            elif close_val < s1_val:
                # Return to S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or return to S1
            if close_val >= entry_price + 1.5 * (prices['high'].iloc[i] - prices['low'].iloc[i]):  # Tighter stop
                # Stop loss hit
                signals[i] = 0.0
                position = 0
            elif close_val > s1_val:
                # Return to S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals