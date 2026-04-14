#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Uses 4-hour Donchian channel (20-period) for breakout signals
# Filters by 1-day EMA (50-period) to ensure trend alignment
# Requires volume > 1.5x 20-period EMA for confirmation
# Designed for 15-25 trades/year with strong trend-following edge in both bull and bear markets
# Uses 1h timeframe for precise entry timing while deriving signals from 4h/1d

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper and lower bands (20-period)
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA (50-period) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period EMA
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long signal: price breaks above Donchian upper band in uptrend
        if (position <= 0 and 
            close[i] > donch_high_aligned[i] and 
            close[i] > ema_1d_aligned[i] and  # Uptrend filter
            volume_confirm):
            position = 1
            signals[i] = position_size
        
        # Short signal: price breaks below Donchian lower band in downtrend
        elif (position >= 0 and 
              close[i] < donch_low_aligned[i] and 
              close[i] < ema_1d_aligned[i] and  # Downtrend filter
              volume_confirm):
            position = -1
            signals[i] = -position_size
        
        # Exit signals: reverse position when opposite breakout occurs
        elif position == 1 and close[i] < donch_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_1dEMA_Volume"
timeframe = "1h"
leverage = 1.0