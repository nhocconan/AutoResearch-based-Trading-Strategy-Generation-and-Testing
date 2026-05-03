#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and 1d volume spike.
# Long when price breaks above 20-period Donchian high AND 12h close > 12h EMA50 (uptrend) AND 1d volume > 2.0x 20-period volume MA.
# Short when price breaks below 20-period Donchian low AND 12h close < 12h EMA50 (downtrend) AND 1d volume > 2.0x 20-period volume MA.
# Position size fixed at 0.25 to balance risk and reward. Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide objective breakout levels, 12h EMA50 filters for trend alignment, 1d volume confirms institutional participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 12h trend when volume confirms.

name = "4h_Donchian20_12hEMA50_1dVolumeSpike"
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
    open_time = prices['open_time']
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_1d = df_1d['volume'].iloc[-1] if len(df_1d) > 0 else 0  # Current 1d volume
        
        # Volume spike condition: current 1d volume > 2.0x 20-period 1d volume MA
        volume_spike = vol_1d > (volume_ma_1d_aligned[i] * 2.0)
        
        # Donchian breakout conditions
        breakout_up = high_val > high_roll[i]  # Price breaks above 20-period high
        breakout_down = low_val < low_roll[i]  # Price breaks below 20-period low
        
        # 12h trend conditions
        trend_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Long: Donchian breakout up AND 12h uptrend AND volume spike
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND 12h downtrend AND volume spike
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Donchian mid-channel OR trend changes
            mid_channel = (high_roll[i] + low_roll[i]) / 2.0
            if close_val < mid_channel or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian mid-channel OR trend changes
            mid_channel = (high_roll[i] + low_roll[i]) / 2.0
            if close_val > mid_channel or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals