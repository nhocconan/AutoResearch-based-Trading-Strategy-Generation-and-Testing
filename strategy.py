#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal + 1d ADX Trend + Volume Confirmation
Hypothesis: In ranging markets (ADX < 25), price tends to reverse at Camarilla H3/L3 levels. In trending markets (ADX >= 25), breakouts at H4/L4 continue. Uses 1d Camarilla pivots for structure and 1d ADX for regime filter. Designed for 6h timeframe to avoid overtrading while capturing both mean reversion and trend continuation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculation: based on previous day's range
    camarilla_h5 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_l5 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_h5[i] = prev_close + 1.5 * range_val
        camarilla_h4[i] = prev_close + 1.25 * range_val
        camarilla_h3[i] = prev_close + 1.125 * range_val
        camarilla_l3[i] = prev_close - 1.125 * range_val
        camarilla_l4[i] = prev_close - 1.25 * range_val
        camarilla_l5[i] = prev_close - 1.5 * range_val
    
    # Calculate 1d ADX(14)
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth TR, +DM, -DM
    atr_1d = np.zeros(len(df_1d))
    plus_di_1d = np.zeros(len(df_1d))
    minus_di_1d = np.zeros(len(df_1d))
    dx_1d = np.zeros(len(df_1d))
    adx_1d = np.zeros(len(df_1d))
    
    # Initial values
    atr_1d[13] = np.mean(tr[1:14])
    plus_dm_14 = np.sum(plus_dm[1:14])
    minus_dm_14 = np.sum(minus_dm[1:14])
    
    for i in range(14, len(df_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
        plus_dm_14 = plus_dm_14 - (plus_dm_14/14) + plus_dm[i]
        minus_dm_14 = minus_dm_14 - (minus_dm_14/14) + minus_dm[i]
        plus_di_1d[i] = 100 * (plus_dm_14 / atr_1d[i]) if atr_1d[i] > 0 else 0
        minus_di_1d[i] = 100 * (minus_dm_14 / atr_1d[i]) if atr_1d[i] > 0 else 0
        dx_1d[i] = (abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i]) * 100) if (plus_di_1d[i] + minus_di_1d[i]) > 0 else 0
    
    # ADX is smoothed DX
    adx_1d[27] = np.mean(dx_1d[14:28])  # First ADX after 2*period
    for i in range(28, len(df_1d)):
        adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14
    
    # Align HTF indicators to LTF
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR(14) for 6h dynamic sizing
    atr_6h = np.full(n, np.nan)
    tr_6h = np.zeros(n)
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_6h[i] = np.mean(tr_6h[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for ADX, ATR, volume MA
    start_idx = max(30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        adx_val = adx_1d_aligned[i]
        atr_val = atr_6h[i]
        vol_ma = vol_ma_20[i]
        
        # Regime filter: ADX < 25 = ranging (mean reversion), ADX >= 25 = trending (breakout)
        ranging = adx_val < 25
        trending = adx_val >= 25
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        volume_confirm = curr_volume > 1.3 * vol_ma
        
        if position == 0:
            if ranging and volume_confirm:
                # Mean reversion: fade at H3/L3
                long_setup = (curr_close <= h3) and (curr_close > l3)
                short_setup = (curr_close >= l3) and (curr_close < h3)
                # More precise: long near L3, short near H3
                long_entry = (curr_close <= l3 * 1.005) and (curr_close >= l3 * 0.995)
                short_entry = (curr_close >= h3 * 0.995) and (curr_close <= h3 * 1.005)
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry = curr_close
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                    lowest_since_entry = curr_close
                else:
                    signals[i] = 0.0
            elif trending and volume_confirm:
                # Breakout: continue at H4/L4
                long_breakout = curr_close > h4
                short_breakout = curr_close < l4
                
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry = curr_close
                elif short_breakout:
                    signals[i] = -0.25
                    position = -1
                    lowest_since_entry = curr_close
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions
            if ranging:
                # In ranging market, exit at opposite H3/L3 or 1.5*ATR profit
                if curr_close >= h3 or curr_close >= (close[i-1] + 1.5 * atr_val):
                    signals[i] = 0.0
                    position = 0
                    highest_since_entry = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In trending market, trail with 2*ATR or exit if trend weakens (ADX falling)
                if curr_close < (highest_since_entry - 2.0 * atr_val) or adx_val < adx_1d_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                    highest_since_entry = 0.0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position management
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions
            if ranging:
                # In ranging market, exit at opposite H3/L3 or 1.5*ATR profit
                if curr_close <= l3 or curr_close <= (close[i-1] - 1.5 * atr_val):
                    signals[i] = 0.0
                    position = 0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In trending market, trail with 2*ATR or exit if trend weakens
                if curr_close > (lowest_since_entry + 2.0 * atr_val) or adx_val < adx_1d_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_Pivot_Reversal_1dADX_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0