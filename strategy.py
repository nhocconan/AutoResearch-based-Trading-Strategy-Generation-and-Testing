#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h strategy using Donchian channel breakouts with volume confirmation and choppiness regime filter.
# Long: price breaks above Donchian(20) high + volume > 1.5x 20-period average + CHOP > 61.8 (range regime = mean reversion setup)
# Short: price breaks below Donchian(20) low + volume > 1.5x 20-period average + CHOP > 61.8
# Exit: opposite Donchian breakout or volume divergence.
# Uses 1d EMA50 for higher timeframe trend filter to avoid extreme counter-trend trades in strong trends.
# Volume confirmation filters weak breakouts. Chop filter ensures we trade in ranging markets where breakouts are more likely to fail and reverse.
# Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
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
    
    # Choppiness Index (14-period) - regime filter
    # CHOP > 61.8 = ranging market (good for mean reversion breakout fades)
    # CHOP < 38.2 = trending market
    true_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    highest_high = high_s.rolling(window=14, min_periods=14).max().values
    lowest_low = low_s.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14.sum() / (highest_high - lowest_low)) / np.log10(14)
    # Fix for division by zero and NaN handling
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # 1d EMA50 for HTF trend filter (avoid trading against strong daily trend)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(chop[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        # HTF trend filter: avoid extreme counter-trend trades
        # In strong uptrend (price > EMA50), avoid shorts
        # In strong downtrend (price < EMA50), avoid longs
        strong_uptrend = close[i] > ema50_1d_aligned[i]
        strong_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR volume divergence
            if close[i] < donchian_low[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR volume divergence
            if close[i] > donchian_high[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above Donchian high + volume confirmed + chop filter + not strong downtrend
            if (close[i] > donchian_high[i] and volume_confirmed and chop_filter and not strong_downtrend):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + volume confirmed + chop filter + not strong uptrend
            elif (close[i] < donchian_low[i] and volume_confirmed and chop_filter and not strong_uptrend):
                position = -1
                signals[i] = -0.25
    
    return signals