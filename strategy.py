#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter and weekly volume confirmation
# Long when price breaks above 6h Donchian upper band AND 1d bullish trend (close > EMA50) AND weekly volume > 1.5x 4-week volume EMA
# Short when price breaks below 6h Donchian lower band AND 1d bearish trend (close < EMA50) AND weekly volume > 1.5x 4-week volume EMA
# Uses 1d EMA50 for trend filter to reduce whipsaw and weekly volume spike for institutional participation confirmation.
# Targets 12-37 trades/year on 6h timeframe with discrete position sizing (0.25) to minimize fee drag.
# Works in bull markets via longs in bullish 1d trend regime and bear markets via shorts in bearish 1d trend regime.
# Weekly volume filter ensures breakouts have sustainable momentum rather than false spikes.

name = "6h_Donchian20_1dTrend_WeeklyVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1d = close_1d > ema_50_1d
    trend_bearish_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 6h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Get weekly data for volume confirmation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Calculate 4-week volume EMA for weekly volume spike filter
    vol_ema_4w = pd.Series(volume_1w).ewm(span=4, adjust=False, min_periods=4).mean().values
    volume_spike_1w = volume_1w > (vol_ema_4w * 1.5)  # Weekly volume at least 1.5x 4-week average
    
    # Align weekly volume spike to 6h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        window_high = high[i - lookback + 1:i + 1]
        window_low = low[i - lookback + 1:i + 1]
        highest_high[i] = np.max(window_high)
        lowest_low[i] = np.min(window_low)
    
    # Calculate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND 1d bullish trend AND weekly volume spike
            if (close[i] > highest_high[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND 1d bearish trend AND weekly volume spike
            elif (close[i] < lowest_low[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band OR 1d trend turns bearish
            if (close[i] < lowest_low[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper band OR 1d trend turns bullish
            if (close[i] > highest_high[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals