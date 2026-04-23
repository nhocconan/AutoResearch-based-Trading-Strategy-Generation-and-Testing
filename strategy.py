#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian breakout with 1w EMA trend filter and ATR-based volatility filter.
- Long: Close breaks above Donchian(20) high + price > 1w EMA50 (uptrend) + ATR(14) > ATR(50) (expanding volatility)
- Short: Close breaks below Donchian(20) low + price < 1w EMA50 (downtrend) + ATR(14) > ATR(50) (expanding volatility)
- Exit: Close crosses Donchian midpoint (mean reversion to center of range)
- Uses 1w HTF for trend direction to avoid counter-trend trades, volatility filter to avoid chop,
  and Donchian breakouts for clear entry/exit levels
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to balance return and minimize fee churn
- Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)
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
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # ATR calculation for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50 and EMA50, 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: ATR(14) > ATR(50) (expanding volatility)
        vol_regime = atr_14[i] > atr_50[i]
        
        if position == 0:
            # Long: Close breaks above Donchian high + uptrend + volatility expansion
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian low + downtrend + volatility expansion
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_VolRegime"
timeframe = "12h"
leverage = 1.0