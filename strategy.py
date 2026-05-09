#!/usr/bin/env python3
# 1D_WeeklyDonchian_Breakout_WickFilter_Volume
# Hypothesis: On daily timeframe, enter long when price breaks above weekly Donchian high with bullish wick (close > open) and volume confirmation.
# Enter short when price breaks below weekly Donchian low with bearish wick (close < open) and volume confirmation.
# Uses weekly trend filter (price > weekly EMA50) to avoid counter-trend trades.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "1D_WeeklyDonchian_Breakout_WickFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get weekly data for Donchian channels and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly Donchian channels (20-period)
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: EMA(50) on close
    ema_50 = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_weekly > ema_50
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align weekly indicators to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    trend_up_aligned = align_htf_to_ltf(prices, df_weekly, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish wick: close > open
        bullish_wick = close[i] > open_price[i]
        # Bearish wick: close < open
        bearish_wick = close[i] < open_price[i]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + bullish wick + weekly uptrend + volume confirmation
            if close[i] > donchian_high_aligned[i] and bullish_wick and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + bearish wick + weekly downtrend + volume confirmation
            elif close[i] < donchian_low_aligned[i] and bearish_wick and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (reversal) or trend changes
            if close[i] < donchian_low_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (reversal) or trend changes
            if close[i] > donchian_high_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals