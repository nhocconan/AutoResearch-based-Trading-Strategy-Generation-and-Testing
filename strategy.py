#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 1d EMA(34) > 1d EMA(89) AND volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian lower band AND 1d EMA(34) < 1d EMA(89) AND volume > 1.5x 20-period average
# Exit when price crosses 4h Donchian middle band (mean reversion) OR 1d EMA crossover reverses
# Uses 4h primary timeframe with 1d HTF for EMA trend filter (more responsive than 1w for regime changes)
# Donchian channels provide clear breakout zones based on recent price extremes
# EMA filter ensures we only trade in trending markets, reducing whipsaw in ranges
# Volume confirmation filters low-momentum breakouts
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_1dEMA_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 90:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) and EMA(89) for trend filter
    close_1d = df_1d['close'].values
    
    # EMA(34)
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # EMA(89)
    ema_89 = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align EMAs to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89)
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20 and len(low) >= 20:
        # Upper band: highest high of last 20 periods
        upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower band: lowest low of last 20 periods
        lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Middle band: average of upper and lower bands
        middle_band = (upper_band + lower_band) / 2
    else:
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        middle_band = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND EMA34 > EMA89 AND volume spike
            if (close[i] > upper_band[i] and 
                ema_34_aligned[i] > ema_89_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND EMA34 < EMA89 AND volume spike
            elif (close[i] < lower_band[i] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle band (mean reversion) OR EMA crossover reverses
            if close[i] < middle_band[i] or ema_34_aligned[i] < ema_89_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle band (mean reversion) OR EMA crossover reverses
            if close[i] > middle_band[i] or ema_34_aligned[i] > ema_89_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals