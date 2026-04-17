#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) Breakout with 1d EMA200 Trend Filter and Volume Spike Confirmation
- Donchian breakout captures momentum in trending markets
- 1d EMA200 filter ensures we only trade in the direction of the long-term trend
- Volume spike (2.0x 20-period average) confirms breakout validity
- ATR-based stoploss (2.5x ATR) manages risk
- Target: 12-37 trades/year per symbol (~50-150 total over 4 years)
- Position sizing: 0.25 (discrete levels to minimize fee churn)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary calculations (Donchian, ATR)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for HTF trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    def calculate_donchian(high_arr, low_arr, window):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper_12h, donchian_lower_12h = calculate_donchian(high_12h, low_12h, 20)
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR (14-period) for stoploss on 12h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_12h = calculate_atr(high_12h, low_12h, close_12h, 14)
    
    # Volume average (20-period) on 12h
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 200  # warmup for EMA200 and other indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema200 = ema200_aligned[i]
        atr_val = atr_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper + above 1d EMA200 + volume spike
            if price > upper and price > ema200 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Donchian lower + below 1d EMA200 + volume spike
            elif price < lower and price < ema200 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price closes below Donchian lower (breakdown)
            if price < lower:
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
            
            # Exit 1: Price closes above Donchian upper (breakout)
            if price > upper:
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

name = "12h_Donchian20_1dEMA200_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0