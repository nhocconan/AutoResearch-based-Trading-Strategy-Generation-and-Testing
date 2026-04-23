#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation + ATR stoploss.
Long when price breaks above Donchian upper AND price > 12h EMA50 AND volume > 1.5x average.
Short when price breaks below Donchian lower AND price < 12h EMA50 AND volume > 1.5x average.
Exit when price crosses Donchian midpoint OR ATR stoploss hit.
Uses discrete position sizing (0.30) to minimize fee churn. Targets 20-50 trades/year per symbol.
Combines price channel breakout with trend filter to work in both trending and ranging markets.
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
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h data
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)))
    tr2 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_4h[0] - low_4h[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 12h EMA50 AND volume confirmation
            if (price > donchian_upper_aligned[i] and 
                price > ema50_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND price < 12h EMA50 AND volume confirmation
            elif (price < donchian_lower_aligned[i] and 
                  price < ema50_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian midpoint
                if price < donchian_mid_aligned[i]:
                    exit_signal = True
                # ATR-based stoploss
                elif price < entry_price - 2.5 * atr_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian midpoint
                if price > donchian_mid_aligned[i]:
                    exit_signal = True
                # ATR-based stoploss
                elif price > entry_price + 2.5 * atr_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0