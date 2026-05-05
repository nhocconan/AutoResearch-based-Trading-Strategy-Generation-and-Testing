#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike
# Long when price breaks above Donchian upper AND close > 12h EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below Donchian lower AND close < 12h EMA50 AND volume > 2.0x 20-period average
# Exit when price crosses 12h EMA50 (trend reversal) OR Donchian middle line
# Uses discrete sizing (0.30) to limit fee drag and manage drawdown
# Target: 100-200 total trades over 4 years (25-50/year) to avoid fee drag while capturing strong trends
# Donchian provides objective breakout levels, 12h EMA50 filters for higher-timeframe trend, volume confirms conviction

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
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
    
    # Donchian channels on 4h data (20-period)
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian upper AND trend filter bullish AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: break below Donchian lower AND trend filter bearish AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: close below 12h EMA50 (trend reversal) OR below Donchian middle (profit taking)
            if close[i] < ema_50_12h_aligned[i] or close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: close above 12h EMA50 (trend reversal) OR above Donchian middle (profit taking)
            if close[i] > ema_50_12h_aligned[i] or close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals