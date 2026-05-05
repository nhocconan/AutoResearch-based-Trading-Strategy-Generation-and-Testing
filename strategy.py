#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian upper (20) AND close > 12h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower (20) AND close < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses Donchian midline (10-period average of high/low) OR trend filter fails
# Uses 4h primary timeframe with 12h HTF for trend filter to capture sustained moves with controlled frequency
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Donchian channels provide clear breakout levels; EMA50 filters for higher-timeframe trend; volume confirms participation

name = "4h_Donchian_20_12hEMA50_Trend_Volume"
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
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels on 4h data
    donchian_window = 20
    if len(high) >= donchian_window:
        donchian_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
        donchian_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND close > 12h EMA50 AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND close < 12h EMA50 AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian middle OR close < 12h EMA50 (trend filter fails)
            if close[i] < donchian_middle[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian middle OR close > 12h EMA50 (trend filter fails)
            if close[i] > donchian_middle[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals