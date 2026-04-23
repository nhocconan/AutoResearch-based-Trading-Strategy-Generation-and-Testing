#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high AND 1w EMA50 uptrend AND volume > 2.0x 20-period average.
Short when price breaks below 20-period Donchian low AND 1w EMA50 downtrend AND volume > 2.0x 20-period average.
Exit when price retouches Donchian midpoint (20-period median) or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.28) to balance return and risk. Targets 12-37 trades/year per symbol.
Donchian channels provide clear breakout structure, 1w EMA50 ensures alignment with weekly trend,
and volume filters reduce false breakouts. Designed for 6h timeframe to minimize fee drag while
capturing multi-day swings. Works in both bull (trend continuation) and bear (mean reversion at extremes).
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian(20) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation (using 6h data)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        ema50 = ema50_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high AND 1w EMA50 uptrend AND volume spike
            if (price > upper and 
                close[i] > ema50 and  # Current close above EMA50 for uptrend
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.28
                position = 1
                entry_price = price
            # Short: Price breaks below Donchian low AND 1w EMA50 downtrend AND volume spike
            elif (price < lower and 
                  close[i] < ema50 and  # Current close below EMA50 for downtrend
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.28
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Donchian midpoint
            if position == 1 and price <= mid:
                exit_signal = True
            elif position == -1 and price >= mid:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.28 if position == 1 else -0.28
    
    return signals

name = "6H_Donchian20_1wEMA50_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0