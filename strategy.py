#!/usr/bin/env python3
"""
exp_6452_12h_donchian20_1d_ema_vol_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Works in bull/bear because Donchian captures breakouts, EMA50 filters trend direction,
volume avoids fakeouts. Discrete position sizing (0.25) minimizes fee churn.
Target trades: 75-150 over 4 years (19-37/year) within proven range.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6452_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Precompute indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index 20 to allow for Donchian calculation
    for i in range(20, n):
        # Skip if EMA not ready
        if ema_50_1d_aligned[i] == 0:
            continue
            
        # Long conditions: price breaks above Donchian high + above 1d EMA50 + volume > average
        long_breakout = close[i] > donchian_high[i-1]  # Use previous bar's high for breakout
        long_trend = close[i] > ema_50_1d_aligned[i]
        long_volume = volume[i] > vol_ma[i]
        
        # Short conditions: price breaks below Donchian low + below 1d EMA50 + volume > average
        short_breakout = close[i] < donchian_low[i-1]  # Use previous bar's low for breakout
        short_trend = close[i] < ema_50_1d_aligned[i]
        short_volume = volume[i] > vol_ma[i]
        
        # ATR-based stoploss (using 14-period ATR)
        if i >= 14:
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = max(tr1, tr2, tr3)
            # Simple ATR approximation using rolling mean
            if i >= 27:  # Need 14+13 for lookback
                atr = np.mean([max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])) 
                              for j in range(i-13, i+1)])
            else:
                atr = 0
            
            # Stoploss: 2 * ATR
            if position == 1 and close[i] < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and close[i] > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry logic
        if position == 0:
            if long_breakout and long_trend and long_volume:
                signals[i] = 0.25  # 25% position
                position = 1
                entry_price = close[i]
            elif short_breakout and short_trend and short_volume:
                signals[i] = -0.25  # 25% short
                position = -1
                entry_price = close[i]
        # Exit logic: reverse signal when opposite breakout occurs
        elif position == 1 and short_breakout and short_trend and short_volume:
            signals[i] = -0.25  # Reverse to short
            position = -1
            entry_price = close[i]
        elif position == -1 and long_breakout and long_trend and long_volume:
            signals[i] = 0.25  # Reverse to long
            position = 1
            entry_price = close[i]
    
    return signals