#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above 6h Donchian upper AND weekly close > weekly EMA50 (bullish regime) AND volume > 1.5x 20 EMA
# Short when price breaks below 6h Donchian lower AND weekly close < weekly EMA50 (bearish regime) AND volume > 1.5x 20 EMA
# Uses 6h for structure, 1w for regime filter to avoid counter-trend trades in bear markets.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-30 trades/year.
# Works in bull markets via longs in bullish regimes and bear markets via shorts in bearish regimes.

name = "6h_Donchian20_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for regime filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for regime filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    bullish_regime = close_1w > ema_50_1w
    bearish_regime = close_1w < ema_50_1w
    
    # Align weekly regime to 6h timeframe
    bullish_regime_aligned = align_htf_to_ltf(prices, df_1w, bullish_regime.astype(float))
    bearish_regime_aligned = align_htf_to_ltf(prices, df_1w, bearish_regime.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    # We need to calculate on 6h data directly
    high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_6h
    donchian_lower = low_6h
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bullish_regime_aligned[i]) or np.isnan(bearish_regime_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND bullish weekly regime AND volume spike
            if (close[i] > donchian_upper[i] and 
                bullish_regime_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND bearish weekly regime AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  bearish_regime_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR weekly regime turns bearish
            if (close[i] < donchian_lower[i] or 
                bearish_regime_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR weekly regime turns bullish
            if (close[i] > donchian_upper[i] or 
                bullish_regime_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals