#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA50 trend filter and volume spike confirmation.
Long when price breaks above 20-period high in 1-day uptrend (close > 1d EMA50) with volume > 2.0x 20-period average.
Short when price breaks below 20-period low in 1-day downtrend (close < 1d EMA50) with volume > 2.0x 20-period average.
Exit via opposite Donchian level (20-period low for longs, 20-period high for shorts).
Designed for ~20-50 trades/year via tight Donchian breakout conditions and volume confirmation.
Uses 1-day trend filter to work in both bull and bear markets, avoiding false breakouts via volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, donchian_period, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above upper Donchian channel with volume confirmation
                long_signal = (close[i] > upper_channel) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below lower Donchian channel with volume confirmation
                short_signal = (close[i] < lower_channel) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian channel
            if close[i] < lower_channel:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian channel
            if close[i] > upper_channel:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0