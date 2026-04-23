#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
Long when price breaks above Donchian high(20) with volume > 1.5x average.
Short when price breaks below Donchian low(20) with volume > 1.5x average.
Exit when price crosses Donchian midpoint (mean of high/low) or ATR stoploss hit.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
Donchian channels provide clear structure, volume confirms breakout strength, ATR stop manages risk.
Works in both bull (breakouts catch trends) and bear (breakdowns catch downtrends) markets.
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
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)))
    tr2 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_4h[0] - low_4h[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators to LTF (15m) - completed bar only
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_4h_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 4h close for price comparison (aligned to LTF)
        price_4h = close_4h[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation
            if (price_4h > donchian_high_aligned[i] and 
                volume_4h[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price_4h
            # Short: price breaks below Donchian low with volume confirmation
            elif (price_4h < donchian_low_aligned[i] and 
                  volume_4h[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price_4h
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian midpoint OR ATR stoploss
                if price_4h < donchian_mid_aligned[i]:
                    exit_signal = True
                elif price_4h < entry_price - 2.5 * atr_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian midpoint OR ATR stoploss
                if price_4h > donchian_mid_aligned[i]:
                    exit_signal = True
                elif price_4h > entry_price + 2.5 * atr_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Volume_ATRStop_MidExit"
timeframe = "4h"
leverage = 1.0