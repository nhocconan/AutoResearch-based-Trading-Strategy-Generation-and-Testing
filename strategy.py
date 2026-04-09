#!/usr/bin/env python3
# 12h_donchian_volume_chop_regime_v1
# Hypothesis: 12h strategy using Donchian channel breakouts with volume confirmation
# and choppiness regime filter. Donchian captures trend breakouts, volume confirms
# momentum strength, and choppiness index avoids false signals in ranging markets.
# Works in both bull and bear markets by filtering entries based on volatility regime.
# Uses 1w HTF for trend alignment and 1d for choppiness calculation.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_volume_chop_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d HTF data for choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian Channel (20-period) on 12h
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness Index (14-period) on 1d
    chop_period = 14
    atr_1d = pd.Series(high_1d - low_1d).rolling(window=1, min_periods=1).sum()  # True Range simplified
    for i in range(1, len(atr_1d)):
        atr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    atr_sum_1d = pd.Series(atr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_1d = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    chop_denominator = np.where((highest_high_1d - lowest_low_1d) == 0, 1, (highest_high_1d - lowest_low_1d))
    chop_value = 100 * np.log10(atr_sum_1d / chop_denominator) / np.log10(chop_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_value)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Choppiness regime: avoid ranging markets (CHOP > 61.8) and extreme trending (CHOP < 38.2)
        # Allow trading only in moderate conditions: 38.2 <= CHOP <= 61.8
        chop_regime = (chop_aligned[i] >= 38.2) and (chop_aligned[i] <= 61.8)
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price breaks above Donchian high AND above 1w EMA (uptrend)
                if close[i] > highest_high[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND below 1w EMA (downtrend)
                elif close[i] < lowest_low[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals