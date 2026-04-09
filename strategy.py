#!/usr/bin/env python3
# 12h_donchian_breakout_volume_chop_v1
# Hypothesis: 12h strategy using Donchian channel breakout with volume confirmation and chop regime filter.
# Long: Price breaks above 20-period Donchian upper + volume > 1.5x 20-period average + chop < 61.8 (trending).
# Short: Price breaks below 20-period Donchian lower + volume > 1.5x 20-period average + chop < 61.8 (trending).
# Exit: Opposite Donchian breakout or chop > 61.8 (range) to avoid whipsaw.
# Uses 1d EMA200 for higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation filters weak breakouts. Chop filter avoids range-bound whipsaw.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) - LTF
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Chop regime filter (14-period) - LTF
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (highest_high - lowest_low)) / np.log10(14)
    
    # 1d EMA200 for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_s_1d = pd.Series(close_1d)
    ema200_1d = close_s_1d.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime filter: chop < 61.8 indicates trending market (avoid range)
        chop_filter = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian lower OR chop > 61.8 (range) OR volume divergence
            if (close[i] < donchian_lower[i] or chop[i] > 61.8 or 
                (close[i] > close[i-1] and volume[i] < volume[i-1])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper OR chop > 61.8 (range) OR volume divergence
            if (close[i] > donchian_upper[i] or chop[i] > 61.8 or 
                (close[i] < close[i-1] and volume[i] < volume[i-1])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian upper + volume confirmed + chop filter + above 1d EMA200
            if (close[i] > donchian_upper[i] and volume_confirmed and chop_filter and 
                close[i] > ema200_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + volume confirmed + chop filter + below 1d EMA200
            elif (close[i] < donchian_lower[i] and volume_confirmed and chop_filter and 
                  close[i] < ema200_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals