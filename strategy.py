#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volume confirmation.
Long when price breaks above upper Donchian channel AND price > 1d EMA50 AND ATR(14) > 1.5 * ATR(50) (vol expansion).
Short when price breaks below lower Donchian channel AND price < 1d EMA50 AND ATR(14) > 1.5 * ATR(50).
Exit when price reverts to Donchian midpoint OR price crosses 1d EMA50 (trend reversal).
Target: 12-37 trades/year on 12h timeframe to minimize fee drag while capturing strong breakouts in both bull and bear markets.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) and ATR(50) for volume confirmation on 12h timeframe
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    
    tr2 = np.maximum(high - low, np.abs(high - close))
    tr2 = np.maximum(tr2, np.abs(low - close))
    
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr2).rolling(window=50, min_periods=50).mean().values
    
    # Donchian channels (20-period) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        atr14_val = atr14[i]
        atr50_val = atr50[i]
        upper_dc = highest_high[i]
        lower_dc = lowest_low[i]
        mid_dc = donchian_mid[i]
        price = close[i]
        
        # Volume expansion filter: ATR(14) > 1.5 * ATR(50)
        vol_expansion = atr14_val > 1.5 * atr50_val
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 1d EMA50 AND vol expansion
            if (price > upper_dc and price > ema50_val and vol_expansion):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND price < 1d EMA50 AND vol expansion
            elif (price < lower_dc and price < ema50_val and vol_expansion):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to midpoint OR price breaks below 1d EMA50 (trend reversal)
                if price <= mid_dc or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to midpoint OR price breaks above 1d EMA50 (trend reversal)
                if price >= mid_dc or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA50_ATR_VolumeExpansion"
timeframe = "12h"
leverage = 1.0