#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with ATR(14) stoploss and volume confirmation.
Long when price breaks above upper band with volume > 1.5x average.
Short when price breaks below lower band with volume > 1.5x average.
Exit when price touches opposite band or ATR-based stoploss is hit.
Uses 1d for ATR calculation and trend filter (price > SMA50 for long, < SMA50 for short).
Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing.
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
    
    # Get 1d data for ATR and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate ATR(14) on 1d data
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(close)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate SMA50 on 1d for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Calculate volume confirmation (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop_multiplier = 2.0
    
    start_idx = max(lookback, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(sma50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_conf = volume_confirm[i]
        atr_val = atr_1d_aligned[i] * 4  # Scale 1d ATR to 4h (approx)
        sma50 = sma50_1d_aligned[i]
        upper_band = upper[i]
        lower_band = lower[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume confirmation and uptrend
            if price > upper_band and vol_conf and price > sma50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band with volume confirmation and downtrend
            elif price < lower_band and vol_conf and price < sma50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: price touches lower band OR ATR stoploss hit
            stop_price = entry_price - atr_stop_multiplier * atr_val
            if price < lower_band or price < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches upper band OR ATR stoploss hit
            stop_price = entry_price + atr_stop_multiplier * atr_val
            if price > upper_band or price > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0