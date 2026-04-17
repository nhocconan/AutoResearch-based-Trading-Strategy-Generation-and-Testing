#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h EMA34 Trend + Volume Spike
Long: Close breaks above Donchian(20) high + price > 12h EMA34 + volume > 1.5x 4h volume SMA(20)
Short: Close breaks below Donchian(20) low + price < 12h EMA34 + volume > 1.5x 4h volume SMA(20)
Exit: Opposite breakout or trailing stop via ATR(10)*3
Uses price channel breakouts for trend capture, filtered by higher timeframe trend and volume.
Designed to work in trending markets with confirmation from higher timeframe and volume.
Target: 75-200 total trades over 4 years (19-50/year)
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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume SMA(20) for volume filter
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(10) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(34, 20)  # need EMA34 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_sma_4h[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_12h = ema_34_12h_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above Donchian high + above 12h EMA + volume spike
            if price > donch_high and price > ema_12h and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: break below Donchian low + below 12h EMA + volume spike
            elif price < donch_low and price < ema_12h and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            if price > highest_since_entry:
                highest_since_entry = price
            
            # Exit conditions: opposite breakout or ATR trailing stop
            if price < donch_low or price < highest_since_entry - 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            
            # Exit conditions: opposite breakout or ATR trailing stop
            if price > donch_high or price > lowest_since_entry + 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0