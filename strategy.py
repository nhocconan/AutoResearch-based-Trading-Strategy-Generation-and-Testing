#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with 1-day Donchian breakout and volume confirmation.
# The Choppiness Index (CHOP) identifies ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In ranging markets, we mean-revert at Donchian channel boundaries; in trending markets, we follow breakouts.
# Volume > 1.3x the 20-period average confirms institutional participation.
# This regime-adaptive approach works in both bull and bear markets by adjusting strategy to market conditions.
# Target: 20-40 trades per year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for regime and trend filters
    df_1d = get_htf_data(prices, '1d')
    
    # Choppiness Index (14-period) for regime detection
    chop_len = 14
    if len(df_1d) < chop_len:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=chop_len, min_periods=chop_len).mean()
    
    # Highest high and lowest low over chop_len
    hh = pd.Series(df_1d['high']).rolling(window=chop_len, min_periods=chop_len).max()
    ll = pd.Series(df_1d['low']).rolling(window=chop_len, min_periods=chop_len).min()
    
    # Choppiness Index: 100 * log10(sum(tr)/ (hh - ll)) / log10(chop_len)
    chop = 100 * np.log10(tr.sum() / (hh - ll)) / np.log10(chop_len)
    chop_values = chop.values
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # 1d Donchian channel (20 periods) for breakout levels
    donch_len = 20
    if len(df_1d) < donch_len:
        return np.zeros(n)
    
    donch_high = pd.Series(df_1d['high']).rolling(window=donch_len, min_periods=donch_len).max().shift(1).values
    donch_low = pd.Series(df_1d['low']).rolling(window=donch_len, min_periods=donch_len).min().shift(1).values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, chop_len, donch_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_1d_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_1d_aligned[i]
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Ranging market (CHOP > 61.8): mean reversion at Donchian boundaries
            if chop_val > 61.8 and vol_confirmed:
                if close[i] <= donch_low_aligned[i]:
                    position = 1  # Long at support
                    signals[i] = position_size
                elif close[i] >= donch_high_aligned[i]:
                    position = -1  # Short at resistance
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            # Trending market (CHOP < 38.2): follow breakouts
            elif chop_val < 38.2 and vol_confirmed:
                if close[i] > donch_high_aligned[i]:
                    position = 1  # Long breakout
                    signals[i] = position_size
                elif close[i] < donch_low_aligned[i]:
                    position = -1  # Short breakdown
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # Transition zone or no volume: stay flat
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches opposite Donchian band or regime shifts strongly
            if (close[i] >= donch_high_aligned[i] or 
                chop_val < 30.0):  # Strong trend signal exits mean reversion
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches opposite Donchian band or regime shifts strongly
            if (close[i] <= donch_low_aligned[i] or 
                chop_val < 30.0):  # Strong trend signal exits mean reversion
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Choppiness_Donchian_Volume_Regime"
timeframe = "4h"
leverage = 1.0