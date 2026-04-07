#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout strategy with 1d trend filter and volume confirmation
# Uses Donchian channel breakouts from 12h timeframe to capture trends in both bull and bear markets
# - Long: break above 12h Donchian upper channel (20-period high) with 1d uptrend and volume confirmation
# - Short: break below 12h Donchian lower channel (20-period low) with 1d downtrend and volume confirmation
# - Exit: reverse signal or price crosses 12h Donchian midline (mean of upper/lower)
# - Volume confirmation avoids false breakouts
# - Designed for low frequency (target: 12-37 trades/year) to minimize fee impact
# - Works in both bull/bear via trend following aligned with higher timeframe

name = "12h_donchian20_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channel (20-period) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid_1d = (high_max_20 + low_min_20) / 2
    
    # Align 1d Donchian levels to 12h timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    donchian_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
    
    # 12h Donchian channel (20-period) for entry signals
    high_max_12 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_12 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid_12 = (high_max_12 + low_min_12) / 2
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or
            np.isnan(donchian_mid_1d_aligned[i]) or np.isnan(donchian_mid_12[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1d Donchian: price above/below midline
        uptrend = close[i] > donchian_mid_1d_aligned[i]
        downtrend = close[i] < donchian_mid_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > high_max_12[i]
        breakout_down = close[i] < low_min_12[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on reverse signal or when price crosses below 12h Donchian midline
            if breakout_down or close[i] < donchian_mid_12[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse signal or when price crosses above 12h Donchian midline
            if breakout_up or close[i] > donchian_mid_12[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: break above 12h Donchian upper channel with 1d uptrend and volume
            if breakout_up and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: break below 12h Donchian lower channel with 1d downtrend and volume
            elif breakout_down and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals