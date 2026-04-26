#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_ChopFilter_v1
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and choppiness regime filter.
- Long when price breaks above Donchian upper band AND 1d EMA50 uptrend AND chop < 61.8 (trending market)
- Short when price breaks below Donchian lower band AND 1d EMA50 downtrend AND chop < 61.8
- Uses Donchian channels for objective breakout levels
- 1d EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Choppiness filter avoids ranging markets where breakouts fail
- Exit on opposite Donchian band or trend reversal
- Designed for low frequency (target 20-40 trades/year on 4h) to minimize fee drag
- Novelty: Donchian breakout with 1d trend and chop filter - different from saturated variants
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for trend and chop filters (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter (needs completed 1d candle)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate 1d choppiness index (needs completed 1d candle)
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest(high,14) - lowest(low,14))) / log10(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14.rolling(window=14, min_periods=14).sum() / 
                          (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    # Chop filter: 1 if trending (chop < 61.8), 0 if ranging/choppy (chop >= 61.8)
    chop_filter = np.where(chop_aligned < 61.8, 1, 0)
    
    # Calculate Donchian channels on 4h chart (primary timeframe)
    # Using 20-period lookback
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 20 for Donchian)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_1d[i]) or np.isnan(chop_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with trend and chop filter
        if position == 0:
            # Long: Price breaks above Donchian upper AND 1d uptrend AND trending market
            if close[i] > donchian_upper[i] and trend_1d[i] == 1 and chop_filter[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND 1d downtrend AND trending market
            elif close[i] < donchian_lower[i] and trend_1d[i] == -1 and chop_filter[i] == 1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian lower OR 1d trend turns down OR market becomes choppy
            if close[i] < donchian_lower[i] or trend_1d[i] == -1 or chop_filter[i] == 0:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian upper OR 1d trend turns up OR market becomes choppy
            if close[i] > donchian_upper[i] or trend_1d[i] == 1 or chop_filter[i] == 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0