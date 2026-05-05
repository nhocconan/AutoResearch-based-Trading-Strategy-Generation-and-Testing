#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND 1d close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below 20-period Donchian low AND 1d close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses 1d EMA50 (trend reversal)
# Uses 6h primary timeframe with 1d HTF for trend filter and Donchian structure
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag
# Donchian channels provide robust breakout levels; 1d EMA50 filters for higher-timeframe trend; volume confirms participation
# Works in both bull and bear markets by following the 1d trend while capturing 6h momentum

name = "6h_Donchian20_Breakout_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data ONCE before loop for trend filter and Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 1d data (20-period high/low)
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian high and low
    donchian_high_1d = pd.Series(h_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(l_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (wait for 1d bar to close)
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
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
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_1d_aligned[i]) or 
            np.isnan(donchian_low_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 1d close > 1d EMA50 AND volume spike
            if (close[i] > donchian_high_1d_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND 1d close < 1d EMA50 AND volume spike
            elif (close[i] < donchian_low_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend reversal)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend reversal)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals