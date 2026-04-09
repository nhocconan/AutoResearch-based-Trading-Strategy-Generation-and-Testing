#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v2
# Hypothesis: 4h Donchian breakout with volume confirmation and chop regime filter for entries.
# Long: Price breaks above 20-period Donchian high + volume > 1.5x 20-period avg + CHOP > 61.8 (range) for mean reversion setup.
# Short: Price breaks below 20-period Donchian low + volume > 1.5x 20-period avg + CHOP > 61.8 (range) for mean reversion setup.
# Exit: Price returns to opposite Donchian level (long exits below Donchian low, short exits above Donchian high).
# Uses 1d HTF for trend filter: only long when 1d close > 1d EMA50, only short when 1d close < 1d EMA50.
# Target: 20-50 trades/year to minimize fee drag while maintaining edge.
# Chop regime filter (CHOP > 61.8) identifies ranging markets where mean reversion at extremes works.
# Donchian breakouts provide structure, volume confirms participation, 1d EMA50 ensures alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Chop regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First period TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 * 14 / (max_high - min_low + 1e-10)) / np.log10(14)
    chop = np.where((max_high - min_low) == 0, 50, chop)  # Handle division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Chop regime filter: CHOP > 61.8 = ranging market (favor mean reversion)
        chop_ranging = chop[i] > 61.8
        # 1d trend filter: close > EMA50 for uptrend, < EMA50 for downtrend
        trend_1d_up = close[i] > ema_50_1d_aligned[i]  # Using current close for simplicity
        trend_1d_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high with volume, chop ranging, and uptrend
            if (close[i] > donchian_high[i] and    # Break above Donchian high
                volume_confirmed and               # Volume spike
                chop_ranging and                   # Chop ranging (mean reversion setup)
                trend_1d_up):                      # 1d uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low with volume, chop ranging, and downtrend
            elif (close[i] < donchian_low[i] and   # Break below Donchian low
                  volume_confirmed and             # Volume spike
                  chop_ranging and                 # Chop ranging (mean reversion setup)
                  trend_1d_down):                  # 1d downtrend
                position = -1
                signals[i] = -0.25
    
    return signals