#!/usr/bin/env python3
# 4h_1d_KC_Reversal_Signal_1dATR_Trend
# Hypothesis: Combines Keltner Channel reversals on 4h with 1d ATR-based trend filter.
# In trending markets (1d ATR rising), look for mean-reversion entries at KC extremes.
# In ranging markets (1d ATR falling), trade breakouts of KC bands.
# Volume confirmation ensures institutional participation. Designed for low trade frequency.
# Works in bull/bear markets by adapting to volatility regime via ATR trend.

name = "4h_1d_KC_Reversal_Signal_1dATR_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2.0) on 4h
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper = ma + 2 * atr
    lower = ma - 2 * atr
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily ATR trend filter: rising ATR = trending market, falling ATR = ranging
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_rising = atr_1d > atr_ma_1d  # trending market
    
    # Align daily ATR trend to 4h
    atr_rising_aligned = align_htf_to_ltf(prices, df_1d, atr_rising)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if np.isnan(ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(atr_rising_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            if atr_rising_aligned[i]:
                # TRENDING: mean reversion at KC extremes
                if close[i] < lower[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > upper[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # RANGING: breakout of KC bands
                if close[i] > upper[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lower[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses back above midpoint OR ATR regime changes
            if close[i] > ma[i] or not atr_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses back below midpoint OR ATR regime changes
            if close[i] < ma[i] or not atr_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals