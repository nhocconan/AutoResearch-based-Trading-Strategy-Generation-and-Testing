#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
Long when price breaks above 20-period high with volume > 1.8x average.
Short when price breaks below 20-period low with volume > 1.8x average.
Exit when price reverts to 20-period midpoint or ATR stoploss hit.
Uses 1d EMA(50) as trend filter: only long when price > EMA50, short when price < EMA50.
Target: 75-200 total trades over 4 years (19-50/year). Discrete sizing at ±0.25 to minimize fee churn.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        mid = np.full_like(high, np.nan)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
            mid[i] = (upper[i] + lower[i]) / 2.0
        return upper, lower, mid
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Calculate ATR(14) for stoploss
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(close)
        atr = np.zeros_like(close)
        
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's ATR
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate volume confirmation (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema50 = ema_50_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        mid = donchian_mid[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and price > EMA50
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian with volume spike and price < EMA50
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: price reverts to midpoint
            if price <= mid:
                exit_signal = True
            # Exit 2: ATR stoploss (2 * ATR below entry)
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: price reverts to midpoint
            if price >= mid:
                exit_signal = True
            # Exit 2: ATR stoploss (2 * ATR above entry)
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_EMA50Filter_ATRStop"
timeframe = "4h"
leverage = 1.0