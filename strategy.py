#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy with 1w ATR-based regime filter and Donchian breakout
# ATR(14) percentile over 1 year identifies low volatility regimes (breakout prone)
# Donchian(20) breakout captures momentum in low volatility environments
# Works in both bull and bear markets: volatility regime filter adapts to changing market conditions
# Uses 1w ATR percentile for regime detection and daily Donchian for entry timing - avoids overtrading

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE for ATR percentile
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w ATR (14 periods)
    atr_len = 14
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation
    atr = pd.Series(tr).rolling(window=atr_len, min_periods=atr_len).mean().values
    
    # ATR percentile over 1 year (52 weeks)
    lookback = 52
    atr_series = pd.Series(atr)
    atr_percentile = atr_series.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align ATR percentile to 1d timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Calculate Donchian channels (20 periods) on daily data
    donch_len = 20
    highest = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lowest = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, atr_len + lookback + donch_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(highest[i]) or 
            np.isnan(lowest[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Regime filter: low volatility environment (ATR percentile < 30)
        low_vol_regime = atr_percentile_aligned[i] < 30
        
        if position == 0:
            # Enter long: low volatility + Donchian breakout above upper band
            if low_vol_regime and price > highest[i]:
                position = 1
                signals[i] = position_size
            # Enter short: low volatility + Donchian breakdown below lower band
            elif low_vol_regime and price < lowest[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR volatility increases
            midpoint = (highest[i] + lowest[i]) / 2
            if price < midpoint or atr_percentile_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR volatility increases
            midpoint = (highest[i] + lowest[i]) / 2
            if price > midpoint or atr_percentile_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wATR_Percentile_Donchian_Breakout_v1"
timeframe = "1d"
leverage = 1.0