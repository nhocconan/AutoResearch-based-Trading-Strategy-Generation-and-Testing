#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high, weekly close > weekly open (bullish weekly candle), and volume > 1.5x 20-day average.
# Short when price breaks below 20-day low, weekly close < weekly open (bearish weekly candle), and volume > 1.5x 20-day average.
# Uses discrete position size 0.25 to minimize churn. Designed for 1d timeframe to capture multi-day trends.
# Target: 20-30 trades/year per symbol (~80-120 total over 4 years).
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 21:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Weekly trend: bullish if close > open, bearish if close < open
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bearish.astype(float))
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above 20-day high, weekly bullish, and volume confirmation
            if price > high_20[i] and weekly_bullish_aligned[i] > 0.5 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below 20-day low, weekly bearish, and volume confirmation
            elif price < low_20[i] and weekly_bearish_aligned[i] > 0.5 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below 20-day low
            if price < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above 20-day high
            if price > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals