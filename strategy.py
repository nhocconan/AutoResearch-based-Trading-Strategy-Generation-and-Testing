#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 1d EMA(50) trending up AND volume > 2x 20-period average
# Short when price breaks below 4h Donchian lower band AND 1d EMA(50) trending down AND volume > 2x 20-period average
# Exit when price crosses 4h Donchian middle band (mean reversion) OR 1d EMA(50) flattens (slope near zero)
# Uses 4h primary timeframe with 1d HTF for EMA trend filter
# Donchian channels provide clear breakout zones based on price structure
# EMA filter ensures we only trade in strong trending markets, reducing whipsaw
# Volume confirmation filters low-momentum breakouts
# Discrete sizing (0.30) to balance profit potential and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_1dEMA50_Trend_Volume"
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
    open_time = prices['open_time'].values
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d EMA(50) slope for trend strength (using 3-bar difference)
    ema_slope = np.zeros_like(ema_50_1d_aligned)
    ema_slope[3:] = (ema_50_1d_aligned[3:] - ema_50_1d_aligned[:-3]) / 3
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Donchian middle band
    middle_band = (highest_high + lowest_low) / 2
    
    # Volume confirmation: volume > 2x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_slope[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND EMA trending up AND volume spike
            if (close[i] > highest_high[i] and 
                ema_slope[i] > 0.001 and  # EMA rising
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower AND EMA trending down AND volume spike
            elif (close[i] < lowest_low[i] and 
                  ema_slope[i] < -0.001 and  # EMA falling
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR EMA flattens (trend weakening)
            if close[i] < middle_band[i] or abs(ema_slope[i]) < 0.0005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR EMA flattens (trend weakening)
            if close[i] > middle_band[i] or abs(ema_slope[i]) < 0.0005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals