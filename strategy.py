#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA34 trend + volume spike + ATR stoploss.
Long when price breaks above Donchian upper with volume > 1.8x average and 12h EMA34 up.
Short when price breaks below Donchian lower with volume > 1.8x average and 12h EMA34 down.
Exit on Donchian middle line or ATR-based stoploss.
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
    
    # Get 12h data for EMA34 trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian(20) on 4h
    lookback = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    dc_middle = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        window_high = high[i-lookback+1:i+1]
        window_low = low[i-lookback+1:i+1]
        dc_upper[i] = np.max(window_high)
        dc_lower[i] = np.min(window_low)
        dc_middle[i] = (dc_upper[i] + dc_lower[i]) / 2.0
    
    # Calculate ATR(14) for stoploss
    atr_period = 14
    tr = np.zeros(n)
    atr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's ATR
    if n > atr_period:
        atr[atr_period] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period+1, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Volume spike: current volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = max(50, lookback, atr_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or 
            np.isnan(dc_middle[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema_up = ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1] if i > 0 else False
        ema_down = ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1] if i > 0 else False
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and EMA up
            if price > dc_upper[i] and vol_spike and ema_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower with volume spike and EMA down
            elif price < dc_lower[i] and vol_spike and ema_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: price returns to middle line OR ATR stoploss hit
            if price <= dc_middle[i] or price <= entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle line OR ATR stoploss hit
            if price >= dc_middle[i] or price >= entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0