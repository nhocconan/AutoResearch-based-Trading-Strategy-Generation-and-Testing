#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 12h volume spike and chop regime filter.
Long when price breaks above 20-period Donchian high AND 12h volume > 2.0x 24-period average AND chop > 61.8 (range regime).
Short when price breaks below 20-period Donchian low AND 12h volume > 2.0x 24-period average AND chop > 61.8 (range regime).
Exit when price crosses Donchian midpoint (mean reversion) OR ATR-based stoploss hit (2.5 * ATR).
Uses 12h HTF for volume confirmation and chop filter to avoid trend-following whipsaws in choppy markets.
Target: 15-25 trades/year per symbol to minimize fee drag while maintaining edge.
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
    
    # Get 4h data for Donchian calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    def calculate_donchian(high_arr, low_arr, window):
        """Donchian Channels: upper=rolling max(high), lower=rolling min(low)"""
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    # Calculate ATR (14-period) for stoploss on 4h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    # Calculate chop regime (choppiness index) on 12h
    def calculate_chop(high_arr, low_arr, close_arr, window):
        """Choppiness Index: higher values indicate ranging market"""
        # True Range
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over window
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        # Chop formula: 100 * log10(sum(tr) / (hh - ll)) / log10(window)
        # Avoid division by zero
        range_hl = hh - ll
        chop = np.full_like(close_arr, 50.0)  # default to neutral
        mask = (range_hl > 0) & (~np.isnan(tr_sum)) & (~np.isnan(hh)) & (~np.isnan(ll))
        chop[mask] = 100 * np.log10(tr_sum[mask] / range_hl[mask]) / np.log10(window)
        
        return chop
    
    # Get Donchian channels
    donch_upper, donch_lower = calculate_donchian(high_4h, low_4h, 20)
    donch_middle = (donch_upper + donch_lower) / 2
    
    # Get 12h data for volume and chop filter (higher timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate volume average (24-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=24, min_periods=24).mean().values
    
    # Calculate chop regime (14-period) on 12h
    chop_12h = calculate_chop(high_12h, low_12h, close_12h, 14)
    
    # Calculate ATR (14-period) for stoploss on 4h
    atr = calculate_atr(high_4h, low_4h, close_4h, 14)
    
    # Align all indicators to 4h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    donch_middle_aligned = align_htf_to_ltf(prices, df_4h, donch_middle)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 35  # warmup for Donchian, volume MA, chop, and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(donch_middle_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        middle = donch_middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        chop_val = chop_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above upper Donchian + 12h volume spike + chop > 61.8 (range)
            if price > upper and vol > 2.0 * vol_ma and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Breakout below lower Donchian + 12h volume spike + chop > 61.8 (range)
            elif price < lower and vol > 2.0 * vol_ma and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price crosses below Donchian middle (mean reversion)
            if price < middle:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR below entry)
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Price crosses above Donchian middle (mean reversion)
            if price > middle:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR above entry)
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hVolumeSpike_ChopFilter_ATRStop"
timeframe = "4h"
leverage = 1.0