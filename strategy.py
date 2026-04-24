#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d volume spike confirmation and chop regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume average and chop regime filter.
- Camarilla levels: calculated from prior 1d OHLC (H3, L3, H4, L4).
- Entry: Long when price breaks above H3 with volume > 1.5 * 20-period average volume AND chop < 61.8 (trending regime).
         Short when price breaks below L3 with volume > 1.5 * 20-period average volume AND chop < 61.8.
- Exit: Opposite breakout (price crosses below L3 for long, above H3 for short) OR chop > 61.8 (range regime).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla breakouts work in trending markets; chop filter avoids false signals in ranging markets.
- Volume confirmation ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, volume average, and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA and chop
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d Chopiness Index (14-period)
    def choppy_index(high_arr, low_arr, close_arr, period=14):
        """Chopiness Index: measures whether market is choppy (range) or trending."""
        atr_sum = np.zeros(len(close_arr))
        true_range = np.maximum(high_arr - low_arr, 
                               np.maximum(np.abs(high_arr - np.roll(close_arr, 1)), 
                                          np.abs(np.roll(close_arr, 1) - low_arr)))
        true_range[0] = high_arr[0] - low_arr[0]  # First TR
        
        # ATR calculation using Wilder's smoothing
        atr = np.zeros_like(true_range)
        atr[period-1] = np.mean(true_range[:period])  # Seed with simple average
        for i in range(period, len(true_range)):
            atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Chopiness Index formula
        hh = np.maximum.accumulate(high_arr)
        ll = np.minimum.accumulate(low_arr)
        hh_ll_diff = hh - ll
        
        chop = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            if hh_ll_diff[i] > 0 and atr_sum[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / hh_ll_diff[i]) / np.log10(period)
            else:
                chop[i] = 50.0  # Neutral value
        return chop
    
    chop_values = choppy_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # H3, L3, H4, L4 levels
    def calculate_camarilla(high_arr, low_arr, close_arr):
        """Calculate Camarilla pivot levels: H3, L3, H4, L4."""
        # Typical price for prior day
        typical_price = (high_arr + low_arr + close_arr) / 3
        range_val = high_arr - low_arr
        
        # Camarilla formulas
        H3 = close_arr + (range_val * 1.1 / 4)
        L3 = close_arr - (range_val * 1.1 / 4)
        H4 = close_arr + (range_val * 1.1 / 2)
        L4 = close_arr - (range_val * 1.1 / 2)
        
        return H3, L3, H4, L4
    
    H3, L3, H4, L4 = calculate_camarilla(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_chop = chop_aligned[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price crosses below L3 OR chop > 61.8 (range regime)
            if position == 1:
                if curr_close < L3_aligned[i] or curr_chop > 61.8:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above H3 OR chop > 61.8 (range regime)
            elif position == -1:
                if curr_close > H3_aligned[i] or curr_chop > 61.8:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and chop filter
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i]
            
            # Chop filter: chop < 61.8 indicates trending regime (avoid ranging markets)
            chop_filter = curr_chop < 61.8
            
            # Long entry: price breaks above H3
            if curr_high > H3_aligned[i] and volume_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below L3
            elif curr_low < L3_aligned[i] and volume_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0