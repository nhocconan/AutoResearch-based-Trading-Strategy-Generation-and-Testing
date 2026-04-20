#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
# - Long: Price breaks above Donchian(20) high + price > 1d EMA(50) + volume > 1.5x avg volume
# - Short: Price breaks below Donchian(20) low + price < 1d EMA(50) + volume > 1.5x avg volume
# - Exit: Opposite Donchian breakout or price crosses 1d EMA(50)
# - Uses 1d EMA for trend filter to avoid counter-trend trades
# - Volume confirmation reduces false breakouts
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Donchian high and low
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation
    avg_volume = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(avg_volume[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume_4h[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above 1d EMA + volume confirmation
            if close_4h[i] > donch_high[i] and close_4h[i] > ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below 1d EMA + volume confirmation
            elif close_4h[i] < donch_low[i] and close_4h[i] < ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR crosses below 1d EMA
            if close_4h[i] < donch_low[i] or close_4h[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR crosses above 1d EMA
            if close_4h[i] > donch_high[i] or close_4h[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_1dEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0