#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 4h Donchian breakout with volume confirmation and chop regime filter.
# Long: Price breaks above Donchian(20) high + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
# Short: Price breaks below Donchian(20) low + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
# Exit: Price returns to Donchian midpoint or opposite breakout
# Uses 12h HTF for trend filter: only long when 12h close > 12h EMA(20), short when < EMA(20)
# Designed for low trade frequency (<400 total) and works in both bull/bear via regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high,n) - min(low,n))) / log10(n)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar TR
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # sum of TR over 1 period = TR itself
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_numerator = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    chop_denominator = max_high - min_low
    chop = np.where(chop_denominator != 0, 100 * np.log10(chop_numerator / chop_denominator) / np.log10(14), 50)
    
    # Get 12h data for trend filter (EMA 20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(chop[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion breakouts)
        regime_filter = chop[i] > 61.8
        
        # Trend filter from 12h: only long when price above 12h EMA, short when below
        trend_filter_long = close[i] > ema_12h_aligned[i]
        trend_filter_short = close[i] < ema_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian midpoint OR breaks below Donchian low
            if close[i] <= donchian_mid[i] or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian midpoint OR breaks above Donchian high
            if close[i] >= donchian_mid[i] or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high with volume confirmation, regime filter, and trend filter
            if (close[i] > donchian_high[i] and volume_confirmed and regime_filter and trend_filter_long):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low with volume confirmation, regime filter, and trend filter
            elif (close[i] < donchian_low[i] and volume_confirmed and regime_filter and trend_filter_short):
                position = -1
                signals[i] = -0.25
    
    return signals