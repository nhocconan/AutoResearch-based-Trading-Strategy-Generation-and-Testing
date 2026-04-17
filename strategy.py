#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + ATR stoploss.
Long when price breaks above Donchian upper band with volume > 1.3x average and price > 1d EMA34.
Short when price breaks below Donchian lower band with volume > 1.3x average and price < 1d EMA34.
Exit via ATR-based trailing stop (2.5x ATR) or Donchian opposite band touch.
Uses 1d for EMA trend filter, 4h for Donchian/channels/volume/ATR.
Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest = np.zeros_like(close)
    lowest = np.zeros_like(close)
    
    for i in range(n):
        if i < lookback - 1:
            highest[i] = np.nan
            lowest[i] = np.nan
        else:
            highest[i] = np.max(high[i-lookback+1:i+1])
            lowest[i] = np.min(low[i-lookback+1:i+1])
    
    # 4h ATR for stoploss (14-period)
    atr_period = 14
    tr = np.zeros(n)
    atr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    if n > atr_period:
        atr[atr_period] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period+1, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # 4h volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, lookback, atr_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        upper = highest[i]
        lower = lowest[i]
        ema34 = ema34_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and uptrend (price > EMA34)
            if price > upper and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below lower band with volume spike and downtrend (price < EMA34)
            elif price < lower and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            
            # Exit conditions: ATR trailing stop OR price touches opposite band
            long_stop = highest_since_entry - 2.5 * atr_val
            if price <= long_stop or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions: ATR trailing stop OR price touches opposite band
            short_stop = lowest_since_entry + 2.5 * atr_val
            if price >= short_stop or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0