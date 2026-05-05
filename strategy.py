#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Williams Fractals for reversal signals with 12h EMA21 trend filter and volume confirmation
# Long when price breaks above daily bullish fractal AND 12h EMA21 > previous 12h EMA21 (uptrend) AND volume > 1.3 * avg_volume(20) on 4h
# Short when price breaks below daily bearish fractal AND 12h EMA21 < previous 12h EMA21 (downtrend) AND volume > 1.3 * avg_volume(20) on 4h
# Exit when price crosses the 12h EMA21 (dynamic stop/reversal)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams Fractals provide reliable reversal points that work in both bull and bear markets
# 12h EMA21 filter ensures we trade with the intermediate trend, reducing whipsaw
# Volume confirmation (1.3x) validates breakout strength without being too restrictive

name = "4h_DailyWilliamsFractal_12hEMA21_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 completed daily bars for fractals
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals
    # Bearish fractal: high[n-2] < high[n-1] and high[n] < high[n-1] and high[n+1] < high[n-1] and high[n+2] < high[n-1]
    # Bullish fractal: low[n-2] > low[n-1] and low[n] > low[n-1] and low[n+1] > low[n-1] and low[n+2] > low[n-1]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i+1] < high_1d[i-1] and 
            high_1d[i+2] < high_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]  # Place at the center bar
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i+1] > low_1d[i-1] and 
            low_1d[i+2] > low_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]  # Place at the center bar
    
    # Align Williams Fractals to 4h timeframe with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Get 12h data ONCE before loop for EMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need at least 21 completed 12h bars for EMA21
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA21
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily bullish fractal, 12h EMA21 rising (uptrend), volume confirmation, in session
            if (close[i] > bullish_fractal_aligned[i] and 
                ema_21_12h_aligned[i] > ema_21_12h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily bearish fractal, 12h EMA21 falling (downtrend), volume confirmation, in session
            elif (close[i] < bearish_fractal_aligned[i] and 
                  ema_21_12h_aligned[i] < ema_21_12h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 12h EMA21
            if close[i] < ema_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 12h EMA21
            if close[i] > ema_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals