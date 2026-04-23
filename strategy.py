#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter.
- Long: Price breaks above Donchian(20) high + price > 1d EMA50 + ATR(14) < ATR(50) (low volatility regime)
- Short: Price breaks below Donchian(20) low + price < 1d EMA50 + ATR(14) < ATR(50)
- Exit: Opposite Donchian breakout or EMA50 trend flip
- Uses Donchian for structure, EMA50 for HTF trend, ATR ratio for regime filter (avoid high volatility whipsaws)
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) and ATR(50) for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0] - close[0]
    tr3[0] = low[0] - close[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 50)  # Need 50 for EMA50, 20 for Donchian, 50 for ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: ATR(14) < ATR(50) (low volatility environment)
        low_vol_regime = atr_14[i] < atr_50[i]
        
        if position == 0:
            # Long: Donchian breakout above + price > 1d EMA50 + low volatility regime
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below + price < 1d EMA50 + low volatility regime
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  low_vol_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown below OR price < 1d EMA50 (trend flip)
            if (close[i] < lowest_low[i]) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout above OR price > 1d EMA50 (trend flip)
            if (close[i] > highest_high[i]) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_ATR_VolFilter"
timeframe = "4h"
leverage = 1.0