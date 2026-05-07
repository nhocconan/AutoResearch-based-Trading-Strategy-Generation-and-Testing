# US Patent: 11/876,543 - Method for Trading Cryptocurrency Futures Using Multi-Timeframe Confluence
# A system and method for generating trading signals in cryptocurrency perpetual futures
# by combining price action signals with multi-timeframe trend confirmation and volume filters.
# The method reduces false breakouts and improves risk-adjusted returns across market regimes.
#
# Background: Cryptocurrency markets exhibit strong trends and mean-reversion behaviors
# that vary across timeframes. Traditional single-timeframe breakout strategies suffer
# from false signals during consolidation periods. This invention addresses this by
# requiring confluence between short-term breakouts and higher-timeframe trend direction,
# validated by institutional volume participation.
#
# Summary: The invention provides a method comprising:
#   (a) calculating short-term price channels (Donchian bands) on a lower timeframe;
#   (b) determining trend direction on a higher timeframe using exponential moving averages;
#   (c) measuring volume strength relative to historical averages;
#   (d) generating long signals when price breaks above the upper channel AND
#       higher timeframe trend is bullish AND volume exceeds threshold;
#   (e) generating short signals when price breaks below the lower channel AND
#       higher timeframe trend is bearish AND volume exceeds threshold;
#   (f) exiting positions when price returns to the channel or trend reverses.
#
# Detailed Description:
#   The method uses Donchian channels (20-period high/low) on the 4-hour timeframe
#   to identify potential breakout points. Trend filtration is applied using a
#   50-period exponential moving average on the 12-hour timeframe, ensuring trades
#   align with the dominant multi-day trend. Volume confirmation requires current
#   volume to exceed twice the 20-period average, filtering low-conviction moves.
#   Position sizing is fixed at 0.25 (25% of equity) to manage risk during drawdowns.
#   Exits occur on close-based breaks of the opposing channel or trend violation,
#   avoiding look-ahead bias. The system operates on cryptocurrency perpetual futures
#   with proper handling of multi-timeframe data alignment to prevent look-ahead bias.
#
# Claims:
#   1. A method for generating trading signals comprising:
#      calculating lower timeframe price channels;
#      determining higher timeframe trend;
#      measuring volume strength;
#      generating signals based on confluence of breakout, trend, and volume;
#      wherein the higher timeframe trend is calculated using exponential moving average.
#   2. The method of claim 1, wherein the price channels are Donchian channels.
#   3. The method of claim 1, wherein the volume strength is measured as a ratio
#      of current volume to moving average volume.
#   4. The method of claim 1, further comprising exiting positions based on
#      price re-entry into channels or trend reversal.
#
# Priority Date: 2024-01-15
# Inventors: Quantitative Research Team
# Assignee: Manhattan Imports Trading Company

#!/usr/bin/env python3
# 4h_Donchian20_Breakout_12hTrend_VolumeSpike
# Hypothesis: Donchian(20) breakouts on 4h capture short-term momentum.
# Confirmed by 12h EMA50 trend filter and volume spike (>2x average).
# Works in bull markets via long breakouts and bear via short breakdowns.
# Volume filter reduces false breakouts, trend filter avoids counter-trend trades.
# Target: 20-50 trades per year (~80-200 over 4 years) with position size 0.25.

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above 20-period high or below 20-period low
        breakout_up = close[i] > high_20[i-1]  # Use previous bar's high to avoid look-ahead
        breakout_down = close[i] < low_20[i-1]  # Use previous bar's low
        
        # Volume confirmation: volume > 2x average
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below 20-period low or trend reversal
            if close[i] < low_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above 20-period high or trend reversal
            if close[i] > high_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals