#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1-day trend filter, volume confirmation, and ATR-based stop.
- Enter long when price breaks above Donchian(20) high, price > 1-day EMA50, and volume > 1.5x 20-period volume MA
- Enter short when price breaks below Donchian(20) low, price < 1-day EMA50, and volume > 1.5x 20-period volume MA
- Exit when price crosses back through Donchian midpoint or ATR stop triggered
- Fixed position size 0.25 to manage drawdown
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
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
    
    # Get 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # ATR for stop loss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_ma_20.iloc[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        atr_val = atr[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        d_mid = donchian_mid[i]
        ema_val = ema_50_aligned[i]
        
        if position == 0:
            # Look for Donchian breakout with volume confirmation and trend filter
            # Long: price breaks above Donchian high, price > EMA50, volume spike
            if price > d_high and price > ema_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low, price < EMA50, volume spike
            elif price < d_low and price < ema_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: price crosses below Donchian mid OR ATR stop
            exit_signal = False
            if price < d_mid:
                exit_signal = True
            elif price <= entry_price - 2.0 * atr_val:  # ATR stop
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price crosses above Donchian mid OR ATR stop
            exit_signal = False
            if price > d_mid:
                exit_signal = True
            elif price >= entry_price + 2.0 * atr_val:  # ATR stop
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0