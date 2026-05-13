#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation.
# Uses 1-week high/low channels for trend direction and 1-day Donchian breakouts for entry timing.
# Long when price breaks above 1d Donchian(20) high and weekly trend is up; short when breaks below 1d Donchian(20) low and weekly trend is down.
# Includes volume confirmation to avoid false breakouts. Designed for low trade frequency (<25/year) to minimize fee drag.
# Works in both bull and bear markets by following the weekly trend direction.

name = "1d_Donchian20_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Weekly trend: price above/below 20-period EMA
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend_up = close_1w > ema20_1w
    weekly_trend_down = close_1w < ema20_1w
    # Align to daily timeframe (wait for weekly bar to close)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Daily Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Donchian
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Get weekly trend values for today
        trend_up = weekly_trend_up_aligned[i]
        trend_down = weekly_trend_down_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high with weekly uptrend and volume
            if close[i] > donchian_high[i] and trend_up and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with weekly downtrend and volume
            elif close[i] < donchian_low[i] and trend_down and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals