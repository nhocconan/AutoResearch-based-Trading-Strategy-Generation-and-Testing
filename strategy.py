#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + ATR(14) stoploss.
Long when price breaks above Donchian upper band with volume > 1.5x average and close > 1d EMA34.
Short when price breaks below Donchian lower band with volume > 1.5x average and close < 1d EMA34.
Exit when price reverts to Donchian midpoint or ATR-based stoploss hit.
Uses 1d for EMA trend filter, 4h for price/volume/Donchian/ATR.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        mid = np.full_like(high, np.nan)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
            mid[i] = (upper[i] + lower[i]) / 2.0
        
        return upper, lower, mid
    
    upper_4h, lower_4h, mid_4h = calculate_donchian(high, low, 20)
    
    # Calculate 4h ATR(14) for stoploss
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
    
    atr_4h = calculate_atr(high, low, close, 14)
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_multiplier = 2.0  # ATR multiplier for stoploss
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_4h[i]) or 
            np.isnan(lower_4h[i]) or 
            np.isnan(mid_4h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema_trend = ema34_1d_aligned[i]
        atr_val = atr_4h[i]
        upper = upper_4h[i]
        lower = lower_4h[i]
        mid = mid_4h[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and bullish trend
            if price > upper and vol_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band with volume spike and bearish trend
            elif price < lower and vol_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: price returns to midpoint OR stoploss hit
            stoploss_price = entry_price - atr_multiplier * atr_val
            if price <= mid or price <= stoploss_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR stoploss hit
            stoploss_price = entry_price + atr_multiplier * atr_val
            if price >= mid or price >= stoploss_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0