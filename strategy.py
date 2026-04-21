#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_ATRFilter_Tight
Hypothesis: Donchian(20) breakout with volume confirmation and ATR-based stoploss on 4h timeframe.
Works in bull/bear: Breakouts capture strong moves in both directions. Volume filter ensures participation.
ATR stoploss limits drawdown during sideways/false breakouts. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20): highest high and lowest low of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss and position sizing
    tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
    tr2 = np.abs(pd.Series(high).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
    tr3 = np.abs(pd.Series(low).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: price breaks above Donchian upper band + volume + 1d uptrend
            if (price > highest_high[i] and 
                volume_ok and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band + volume + 1d downtrend
            elif (price < lowest_low[i] and 
                  volume_ok and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stoploss or trend reversal
            stop_price = entry_price - 2.5 * atr[i]
            if price < stop_price or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stoploss or trend reversal
            stop_price = entry_price + 2.5 * atr[i]
            if price > stop_price or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATRFilter_Tight"
timeframe = "4h"
leverage = 1.0