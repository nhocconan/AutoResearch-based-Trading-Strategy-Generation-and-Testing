#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Fractals with 1w EMA200 trend filter and volume confirmation
# Long when price breaks above 1d bearish fractal (swing high) AND 1w EMA200 > EMA200 previous (uptrend) AND volume > 1.3 * avg_volume(50) on 12h
# Short when price breaks below 1d bullish fractal (swing low) AND 1w EMA200 < EMA200 previous (downtrend) AND volume > 1.3 * avg_volume(50) on 12h
# Exit when price crosses the 1d midline (average of recent swing high/low)
# Uses discrete sizing 0.28 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Fractals provide strong swing points that reduce whipsaw
# 1w EMA200 trend filter ensures we trade with the dominant weekly trend
# Volume confirmation (1.3x) validates breakout strength while limiting overtrading

name = "12h_1dWilliamsFractal_1wEMA200_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 completed 1d bars for fractals
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals: bearish (swing high) and bullish (swing low)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal (swing high)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i-1] > high_1d[i+2] and 
            high_1d[i-2] > high_1d[i+1]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal (swing low)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i-1] < low_1d[i+2] and 
            low_1d[i-2] < low_1d[i+1]):
            bullish_fractal[i] = low_1d[i]
    
    # Align 1d Williams Fractals to 12h timeframe with additional delay for confirmation
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d midline (average of recent swing high/low for exit)
    # Use rolling window to find recent swing high/low
    swing_high = pd.Series(bearish_fractal).rolling(window=50, min_periods=1).max().values
    swing_low = pd.Series(bullish_fractal).rolling(window=50, min_periods=1).min().values
    midline_1d = (swing_high + swing_low) / 2.0
    midline_aligned = align_htf_to_ltf(prices, df_1d, midline_1d)
    
    # Get 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need at least 200 completed weekly bars for EMA200
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate volume confirmation: volume > 1.3 * 50-period average volume on 12h
    avg_volume_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.3 * avg_volume_50)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(150, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(midline_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(avg_volume_50[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d bearish fractal (swing high), 1w EMA200 > EMA200 previous (uptrend), volume confirmation, in session
            if (close[i] > bearish_fractal_aligned[i] and 
                ema_200_1w_aligned[i] > ema_200_1w_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.28
                position = 1
            # Short: price breaks below 1d bullish fractal (swing low), 1w EMA200 < EMA200 previous (downtrend), volume confirmation, in session
            elif (close[i] < bullish_fractal_aligned[i] and 
                  ema_200_1w_aligned[i] < ema_200_1w_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d midline
            if close[i] < midline_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price crosses back above 1d midline
            if close[i] > midline_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals