#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above upper band AND close > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below lower band AND close < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses 1d EMA34 (trend change)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years) to avoid overtrading.
# Donchian channels provide structural breakouts with clear levels.
# Volume spike confirms institutional participation, reducing false breakouts.
# 1d EMA34 trend filter ensures alignment with long-term direction, working in both bull and bear regimes.

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels from 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2.0x 20-bar average volume (strict to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Donchian and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_upper = upper[i]
        curr_lower = lower[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA34 (trend change)
            if curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA34 (trend change)
            if curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND close > 1d EMA34 AND volume confirmation
            if curr_close > curr_upper and curr_close > curr_ema34_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower band AND close < 1d EMA34 AND volume confirmation
            elif curr_close < curr_lower and curr_close < curr_ema34_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals