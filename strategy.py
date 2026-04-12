#!/usr/bin/env python3
"""
6h_1d_limited_range_breakout_v1
Hypothesis: In BTC/ETH, price often consolidates in tight ranges (low volatility) before explosive moves.
We detect tight ranges using 24-period ATR% (ATR/Close) on 1d timeframe. When volatility contracts below
the 20th percentile, we wait for a breakout of the prior 20-period high/low on 6s timeframe with volume
confirmation. This avoids whipsaws in high volatility and captures momentum after consolidation.
Works in bull/bear because it trades breakouts regardless of direction, using volatility regime filter.
Target: 20-40 trades/year (80-160 total over 4 years).
"""

name = "6h_1d_limited_range_breakout_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for volatility and range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 24-period ATR on daily
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_24 = pd.Series(tr).rolling(window=24, min_periods=24).mean().values
    
    # ATR as percentage of price (volatility measure)
    atr_pct = atr_24 / close_1d
    
    # 20-period high/low for breakout levels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volatility regime: low volatility when ATR% < 20th percentile (lookback 100 days)
    atr_pct_series = pd.Series(atr_pct)
    vol_percentile = atr_pct_series.rolling(window=100, min_periods=50).quantile(0.20).values
    low_vol_regime = atr_pct < vol_percentile
    
    # Align all daily indicators to 6s timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    # Volume confirmation on 6s: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(low_vol_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: break above 20-period high in low volatility regime with volume
        if (close[i] > high_20_aligned[i] and low_vol_aligned[i] > 0.5 and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: break below 20-period low in low volatility regime with volume
        elif (close[i] < low_20_aligned[i] and low_vol_aligned[i] > 0.5 and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or volatility expands (exit consolidation breakout)
        elif position == 1 and (low_vol_aligned[i] < 0.5 or close[i] < low_20_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low_vol_aligned[i] < 0.5 or close[i] > high_20_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals