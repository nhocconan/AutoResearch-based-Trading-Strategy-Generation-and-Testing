#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike_Regime
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation (2.0x) plus choppiness regime filter.
Optimized for 1d timeframe to target 30-100 trades over 4 years (7-25/year) by using tight entry conditions and discrete position sizing (0.30).
Works in bull/bear via weekly trend alignment and choppiness filter to avoid whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate Donchian channels (20-period) on daily data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2.0 * volume_ma(50) for strong confirmation
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Choppiness regime filter (14-period) - avoid trading in choppy markets
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.append(close[0], close[:-1]))),
                                  np.abs(low - np.append(close[0], close[:-1])))).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr_14 * 14) / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_regime = chop < 61.8  # Trending regime (below 61.8 = trending, above = ranging)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, 20 for Donchian, 50 for volume MA, 14 for chop)
    start_idx = max(50, 20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Donchian breakout conditions with volume, trend, and regime confirmation
        if position == 0:
            # Long: Price breaks above Donchian upper band AND weekly uptrend AND volume spike AND trending regime
            if close[i] > highest_high[i] and trend_1w[i] == 1 and volume_spike[i] and chop_regime[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below Donchian lower band AND weekly downtrend AND volume spike AND trending regime
            elif close[i] < lowest_low[i] and trend_1w[i] == -1 and volume_spike[i] and chop_regime[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: Price falls below Donchian lower band OR weekly trend turns down
            if close[i] < lowest_low[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: Price rises above Donchian upper band OR weekly trend turns up
            if close[i] > highest_high[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_Regime"
timeframe = "1d"
leverage = 1.0