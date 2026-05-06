#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA200 trend filter and volume confirmation
# Long when price breaks above upper Donchian(20) AND close > 12h EMA200 (uptrend) AND volume > 1.8 * 20-bar avg volume
# Short when price breaks below lower Donchian(20) AND close < 12h EMA200 (downtrend) AND volume > 1.8 * 20-bar avg volume
# Exit when price retraces to 50% of the Donchian channel width from the breakout level
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h EMA200 provides strong long-term trend filter to avoid counter-trend trades
# Volume spike threshold optimized to reduce false breakouts while maintaining sufficient trade frequency
# 50% retracement exit captures mean reversion in ranging markets and protects profits in trends

name = "4h_Donchian20_12hEMA200_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels for 4h timeframe (based on previous 20 bars)
    lookback = 20
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    channel_mid = (upper_channel + lower_channel) / 2.0
    channel_width = upper_channel - lower_channel
    
    # Get 12h data ONCE before loop for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA200
    close_12h_series = pd.Series(close_12h)
    ema200_12h = close_12h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate volume confirmation: volume > 1.8 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema200_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper channel AND uptrend AND volume spike
            if close[i] > upper_channel[i] and close[i] > ema200_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower channel AND downtrend AND volume spike
            elif close[i] < lower_channel[i] and close[i] < ema200_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to 50% of channel width from upper channel
            exit_level = upper_channel[i] - (0.5 * channel_width[i])
            if close[i] <= exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to 50% of channel width from lower channel
            exit_level = lower_channel[i] + (0.5 * channel_width[i])
            if close[i] >= exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals