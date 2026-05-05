#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 1w close > 1w EMA50 AND volume > 1.8x 20-period average
# Short when price breaks below 4h Donchian lower band AND 1w close < 1w EMA50 AND volume > 1.8x 20-period average
# Exit when price crosses 1w EMA50 (trend reversal) OR price retouches Donchian middle band (mean reversion)
# Uses 4h primary timeframe with 1w HTF for trend filter
# Donchian provides clear breakout levels, volume confirmation reduces false signals, weekly EMA filters counter-trend noise
# Discrete sizing (0.25) to limit fee drag and manage drawdown in bear markets
# Target: 75-150 total trades over 4 years (19-37/year) for 4h timeframe

name = "4h_Donchian20_Breakout_1wEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20:
        # Donchian upper band: 20-period high
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian lower band: 20-period low
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Donchian middle band: (upper + lower) / 2
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.8x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 1w close > 1w EMA50 AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 1w close < 1w EMA50 AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA50 (trend reversal) OR price retouches Donchian middle (mean reversion)
            if close[i] < ema_50_1w_aligned[i] or abs(close[i] - donchian_middle[i]) < 0.001 * donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA50 (trend reversal) OR price retouches Donchian middle (mean reversion)
            if close[i] > ema_50_1w_aligned[i] or abs(close[i] - donchian_middle[i]) < 0.001 * donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals