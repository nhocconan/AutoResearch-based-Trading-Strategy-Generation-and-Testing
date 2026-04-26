#!/usr/bin/env python3
"""
1d_Donchian_20_Breakout_1wTrend_VolumeSpike_RegimeFilter
Hypothesis: On daily timeframe, use 20-day Donchian channel breakouts with weekly trend filter (close > weekly EMA20) and volume confirmation (>1.5x 20-day average volume). Add choppiness regime filter (CHOP > 61.8 = range, only mean-revert at extremes) to avoid whipsaws in bear markets. Target: 15-25 trades/year to stay within fee limits while capturing major trends. Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 periods for weekly EMA20
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA20 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average volume
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending
    def choppiness_index(high, low, close, window=14):
        # True range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for first bar
        
        # ATR = smoothed TR (using Wilder's smoothing = EMA with alpha=1/window)
        atr = pd.Series(tr).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        
        # Max(high) - Min(low) over window
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        hh_ll = max_high - min_low
        
        # CHOP = 100 * log10(sum(atr)/hh_ll) / log10(window)
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        chop = 100 * np.log10(sum_atr / np.maximum(hh_ll, 1e-10)) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    chop_range = chop > 61.8  # range regime
    chop_trend = chop < 38.2  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) + weekly EMA20 + volume MA + chop warmup
    start_idx = max(20, 20, 14)  # 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend alignment
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume spike + NOT in strong range
            long_breakout = close[i] > donchian_high[i]
            long_signal = long_breakout and weekly_uptrend and volume_spike[i] and not chop_range[i]
            
            # Short: price breaks below Donchian low + weekly downtrend + volume spike + NOT in strong range
            short_breakout = close[i] < donchian_low[i]
            short_signal = short_breakout and weekly_downtrend and volume_spike[i] and not chop_range[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Donchian low OR weekly trend turns down OR chop becomes strong range
            if (close[i] < donchian_low[i] or not weekly_uptrend or chop_range[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Donchian high OR weekly trend turns up OR chop becomes strong range
            if (close[i] > donchian_high[i] or not weekly_downtrend or chop_range[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian_20_Breakout_1wTrend_VolumeSpike_RegimeFilter"
timeframe = "1d"
leverage = 1.0