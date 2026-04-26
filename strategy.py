#!/usr/bin/env python3
"""
1d_Regime_Adaptive_Donchian_Volume
Hypothesis: Daily Donchian(20) breakout with volume confirmation and weekly chop regime filter.
In trending markets (CHOP < 38.2): follow breakouts. In ranging markets (CHOP > 61.8): fade extremes.
Works in both bull/bear by adapting to market regime. Designed for 30-100 total trades over 4 years.
"""

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly chop regime: CHOP(14) = 100 * log10(sum(ATR(1)) / (max(high)-min(low))) / log10(14)
    tr_1w = np.maximum(np.maximum(df_1w['high'].values - df_1w['low'].values,
                                  np.abs(df_1w['high'].values - np.concatenate([[df_1w['close'][0]], df_1w['close'][:-1]])),
                                  np.abs(df_1w['low'].values - np.concatenate([[df_1w['close'][0]], df_1w['close'][:-1]]))))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    sum_atr_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    max_high_1w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_1w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_1w - min_low_1w
    chop_1w = np.where(chop_denominator > 0, 100 * np.log10(sum_atr_1w / chop_denominator) / np.log10(14), 50)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-day EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        chop = chop_1w_aligned[i]
        
        # Regime logic
        if chop < 38.2:  # Trending regime - follow breakouts
            # Long: break above Donchian high + volume spike
            if close[i] > donchian_high[i] and volume_spike[i]:
                if position != 1:
                    signals[i] = base_size
                    position = 1
                else:
                    signals[i] = base_size
            # Short: break below Donchian low + volume spike
            elif close[i] < donchian_low[i] and volume_spike[i]:
                if position != -1:
                    signals[i] = -base_size
                    position = -1
                else:
                    signals[i] = -base_size
            # Exit: reverse signal
            elif position == 1 and close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = base_size
                else:
                    signals[i] = -base_size
                    
        elif chop > 61.8:  # Ranging regime - fade extremes
            # Long: pullback to Donchian low + volume spike (mean reversion)
            if close[i] <= donchian_low[i] * 1.001 and volume_spike[i]:  # Allow small tolerance
                if position != 1:
                    signals[i] = base_size
                    position = 1
                else:
                    signals[i] = base_size
            # Short: pullback to Donchian high + volume spike
            elif close[i] >= donchian_high[i] * 0.999 and volume_spike[i]:
                if position != -1:
                    signals[i] = -base_size
                    position = -1
                else:
                    signals[i] = -base_size
            # Exit: move toward middle of range
            elif position == 1 and close[i] >= (donchian_high[i] + donchian_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] <= (donchian_high[i] + donchian_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = base_size
                else:
                    signals[i] = -base_size
        else:  # Neutral regime (38.2 <= CHOP <= 61.8) - no trades
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1d_Regime_Adaptive_Donchian_Volume"
timeframe = "1d"
leverage = 1.0