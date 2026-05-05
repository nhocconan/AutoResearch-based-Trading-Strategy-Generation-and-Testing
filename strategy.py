#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above upper band AND close > EMA50(1d) AND volume > 1.5x 20-period average
# Short when price breaks below lower band AND close < EMA50(1d) AND volume > 1.5x 20-period average
# Exit when price crosses back to opposite Donchian band OR EMA50(1d) trend flips
# Uses 12h primary timeframe for lower trade frequency (target: 12-37/year)
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Donchian provides structure, EMA50 filters trend, volume confirms conviction

name = "12h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    if len(high) >= 20:
        upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND close > EMA50(1d) AND volume spike
            if (close[i] > upper_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND close < EMA50(1d) AND volume spike
            elif (close[i] < lower_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower band OR close < EMA50(1d) (trend flip)
            if (close[i] < lower_band[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper band OR close > EMA50(1d) (trend flip)
            if (close[i] > upper_band[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals