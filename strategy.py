#!/usr/bin/env python3
# 1d_weekly_volume_regime_v1
# Hypothesis: 1d strategy using 1w trend filter + daily volume spike + Donchian breakout.
# Long: Price breaks above 20-day Donchian high + volume > 1.5x 20-day average + weekly close > weekly open (bullish week)
# Short: Price breaks below 20-day Donchian low + volume > 1.5x 20-day average + weekly close < weekly open (bearish week)
# Exit: Price closes back inside 10-day Donchian channel (reduces whipsaw)
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong momentum moves in both bull and bear markets while avoiding chop.
# Target: 15-25 trades/year (60-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly trend: bullish if weekly close > weekly open, bearish if weekly close < weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly trend to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Daily Donchian channels (20-period for entry, 10-period for exit)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high_20 = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_s.rolling(window=20, min_periods=20).min().values
    donchian_high_10 = high_s.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_s.rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price closes back below 10-day Donchian low (reduces whipsaw)
            if close[i] < donchian_low_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back above 10-day Donchian high (reduces whipsaw)
            if close[i] > donchian_high_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and weekly trend filter
            bullish_breakout = (close[i] > donchian_high_20[i]) and volume_confirmed and weekly_bullish_aligned[i] > 0.5
            bearish_breakout = (close[i] < donchian_low_20[i]) and volume_confirmed and weekly_bearish_aligned[i] > 0.5
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals