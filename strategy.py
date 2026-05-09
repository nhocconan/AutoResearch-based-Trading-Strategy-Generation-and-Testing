#!/usr/bin/env python3
# 1D_WeeklyDonchian_Breakout_WickFilter_Volume
# Hypothesis: On 1d timeframe, enter long when price breaks above weekly Donchian high with bullish candle close and volume confirmation.
# Enter short when price breaks below weekly Donchian low with bearish candle close and volume confirmation.
# Weekly trend filter avoids counter-trade trades. Target: 10-25 trades/year per symbol (40-100 total over 4 years).
# Works in bull via breakouts, in bear via short breakdowns with trend filter.

name = "1D_WeeklyDonchian_Breakout_WickFilter_Volume"
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
    
    # Get weekly data for Donchian channels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: EMA(34) on close
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1w > ema_34
    
    # Daily volume confirmation: current volume > 1.5x 20-day average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Daily bullish/bearish candle: close > open for bull, close < open for bear
    open_prices = prices['open'].values
    bullish_candle = close > open_prices
    bearish_candle = close < open_prices
    
    # Align weekly indicators to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + bullish candle + weekly uptrend + volume confirmation
            if close[i] > donchian_high_aligned[i] and bullish_candle[i] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + bearish candle + weekly downtrend + volume confirmation
            elif close[i] < donchian_low_aligned[i] and bearish_candle[i] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low or trend changes to down
            if close[i] < donchian_low_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high or trend changes to up
            if close[i] > donchian_high_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals