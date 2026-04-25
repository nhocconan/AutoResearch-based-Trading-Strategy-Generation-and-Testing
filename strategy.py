#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dRegime_ChopFilter_v1
Hypothesis: Trade Donchian(20) breakouts on 4h timeframe with 1-day choppiness regime filter.
In trending markets (CHOP < 38.2): trade breakouts in direction of trend.
In ranging markets (CHOP > 61.8): fade breakouts (mean reversion).
Requires volume > 1.3x 20-period average for confirmation.
Position size: 0.25 to limit drawdown and reduce fee churn.
Target: 80-180 total trades over 4 years = 20-45/year.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for chop calculation
        return np.zeros(n)
    
    # Calculate 1-day choppiness index: CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(N)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr1 = tr  # ATR(1) is just TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    chop_raw = 100 * np.log10(sum_atr1 / range_14) / np.log10(14)
    chop_1d = chop_raw  # already aligned to 1d bars
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 20-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Donchian channels on 4h
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), chop (14+14), volume MA (20)
    start_idx = max(donchian_window, 28, 20)  # 28 for chop (14+14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        chop = chop_aligned[i]
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
            is_trending = chop < 38.2
            is_ranging = chop > 61.8
            
            if is_trending and volume_confirm:
                # Trending market: trade breakouts in direction of momentum
                long_setup = (close[i] > highest_high[i])
                short_setup = (close[i] < lowest_low[i])
                
                if long_setup:
                    signals[i] = 0.25
                    position = 1
                elif short_setup:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    
            elif is_ranging and volume_confirm:
                # Ranging market: fade breakouts (mean reversion)
                # Long when price breaks below lower band (oversold bounce)
                # Short when price breaks above upper band (overbought reversal)
                long_setup = (close[i] < lowest_low[i])
                short_setup = (close[i] > highest_high[i])
                
                if long_setup:
                    signals[i] = 0.25
                    position = 1
                elif short_setup:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Choppy middle zone or no volume confirmation: stay flat
                signals[i] = 0.0
                
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches opposite Donchian band OR regime shifts to ranging
            if (close[i] <= lowest_low[i]) or (chop > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches opposite Donchian band OR regime shifts to ranging
            if (close[i] >= highest_high[i]) or (chop > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dRegime_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0