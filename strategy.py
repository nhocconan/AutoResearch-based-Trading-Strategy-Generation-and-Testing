#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and ATR regime filter.
- Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x ATR(10)*close (volatility regime) AND ATR(14) > ATR(50) (trending regime)
- Short when price breaks below Camarilla L3 level AND same filters
- Exits when price reverts to Camarilla H4/L4 levels or ATR regime shifts to choppy
- Uses 6h primary timeframe with 1d HTF for pivot calculation and volume/ATR regime
- Camarilla levels from 1d provide intraday support/resistance that work in both trending and ranging markets
- Volume spike filter ensures breakouts occur with conviction
- ATR regime filter avoids false breakouts in choppy markets
- Designed for BTC/ETH with edge in breakout continuations and mean reversion at extremes
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
    
    # Get 1d data ONCE before loop for Camarilla pivots, volume, and ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d volume and ATR for regime filters
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(10) for volume threshold
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = np.nan
    low_close_1d[0] = np.nan
    tr_1d = np.maximum(np.maximum(high_low_1d, high_close_1d), low_close_1d)
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # 1d ATR(50) for regime filter (trending vs choppy)
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Volume spike condition: 1d volume > 1.5 * ATR(10) * close (volatility-adjusted)
    vol_threshold_1d = 1.5 * atr_10_1d_aligned * close
    volume_spike = volume_1d > vol_threshold_1d  # This is already aligned via align_htf_to_ltf
    
    # ATR regime: trending when ATR(14) > ATR(50) (using 1d ATR)
    atr_regime_trending = atr_10_1d_aligned > atr_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 10) + 1  # Need 50 for ATR(50), 10 for ATR(10)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_regime_trending[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3, volume spike, trending regime
            if close[i] > camarilla_h3_aligned[i] and volume_spike[i] and atr_regime_trending[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3, volume spike, trending regime
            elif close[i] < camarilla_l3_aligned[i] and volume_spike[i] and atr_regime_trending[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches Camarilla H4 (take profit) OR regime shifts to choppy
            if close[i] >= camarilla_h4_aligned[i] or not atr_regime_trending[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches Camarilla L4 (take profit) OR regime shifts to choppy
            if close[i] <= camarilla_l4_aligned[i] or not atr_regime_trending[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dVolSpike_ATRRegime_v1"
timeframe = "6h"
leverage = 1.0