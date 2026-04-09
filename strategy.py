#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# In trending markets, breakouts capture momentum; in ranging markets (2025+), chop filter avoids false breakouts.
# Volume confirmation ensures breakout validity. Discrete sizing (0.0, ±0.30) minimizes fee churn.
# Target: 75-200 total trades over 4 years by requiring Donchian breakout + volume spike + chop < 61.8 (trending) or > 61.8 (ranging mean reversion).
# Primary timeframe: 4h, HTF: 1d for regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index: 100 * log10(sum(tr14) / log10(14) * (hh14 - ll14)) / log10(14)
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((hh_14 - ll_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channel (20-period) on 4h
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low or volume dries up
            if close[i] < donchian_low[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high or volume dries up
            if close[i] > donchian_high[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            if volume_confirmed:
                # Regime-based entry logic
                chop_val = chop_aligned[i]
                # Trending regime (chop < 38.2): breakout follow
                if chop_val < 38.2:
                    # Long breakout: price closes above Donchian high
                    if close[i] > donchian_high[i]:
                        position = 1
                        signals[i] = 0.30
                    # Short breakout: price closes below Donchian low
                    elif close[i] < donchian_low[i]:
                        position = -1
                        signals[i] = -0.30
                # Ranging regime (chop > 61.8): mean reversion at extremes
                elif chop_val > 61.8:
                    # Long mean reversion: price touches Donchian low
                    if low[i] <= donchian_low[i]:
                        position = 1
                        signals[i] = 0.30
                    # Short mean reversion: price touches Donchian high
                    elif high[i] >= donchian_high[i]:
                        position = -1
                        signals[i] = -0.30
                # Neutral regime (38.2 <= chop <= 61.8): no trade
    
    return signals