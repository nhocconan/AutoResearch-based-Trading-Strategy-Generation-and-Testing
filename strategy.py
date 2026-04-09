#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long: Price breaks above Donchian(20) upper band, volume > 1.5x 20-period average, and choppy market (CHOP > 61.8).
# Short: Price breaks below Donchian(20) lower band, volume > 1.5x 20-period average, and choppy market (CHOP > 61.8).
# Exit: Opposite Donchian break or chop regime shifts to trending (CHOP < 38.2).
# Uses 1d timeframe for choppiness calculation to avoid look-ahead.
# Target: 20-50 trades/year (75-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on 4h
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for choppiness calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate choppiness index on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(ATR(14) / (max_high_14 - min_low_14)) / log10(14)
    range_14 = max_high_14 - min_low_14
    chop_ratio = np.where(range_14 > 0, atr_14 / range_14, np.nan)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align HTF choppiness to 4h timeframe (wait for completed 1d bar)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion
        choppy_market = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian lower OR chop regime shifts to trending
            if low[i] < donchian_low[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper OR chop regime shifts to trending
            if high[i] > donchian_high[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian upper, volume confirmed, choppy market
            if (high[i] > donchian_high[i] and volume_confirmed and choppy_market):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower, volume confirmed, choppy market
            elif (low[i] < donchian_low[i] and volume_confirmed and choppy_market):
                position = -1
                signals[i] = -0.25
    
    return signals