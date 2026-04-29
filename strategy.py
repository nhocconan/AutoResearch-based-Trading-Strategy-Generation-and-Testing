#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses Donchian channel from 4h for breakout detection with tight 20-period lookback
# 1d EMA34 provides strong trend filter to avoid counter-trend trades in ranging markets
# Volume > 1.8x average confirms institutional participation and reduces false breakouts
# Exit on opposite Donchian(10) touch for quick profit taking and whipsaw reduction
# Discrete position sizing (0.25) designed for ~25-40 trades/year to minimize fee drag
# Works in bull/bear via trend filter - only trades in direction of 1d EMA34

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channels (20 for entry, 10 for exit)
    # Donchian upper (20-period high)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian upper (10-period for exit)
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    # Donchian lower (10-period for exit)
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Donchian20 and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_high_20 = donchian_high_20[i]
        curr_donchian_low_20 = donchian_low_20[i]
        curr_donchian_high_10 = donchian_high_10[i]
        curr_donchian_low_10 = donchian_low_10[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price touches Donchian lower (10-period) for profit taking
            if curr_low <= curr_donchian_low_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches Donchian upper (10-period) for profit taking
            if curr_high >= curr_donchian_high_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above Donchian upper (20) with 1d EMA34 uptrend and volume
            if curr_high > curr_donchian_high_20 and curr_close > curr_ema34_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower (20) with 1d EMA34 downtrend and volume
            elif curr_low < curr_donchian_low_20 and curr_close < curr_ema34_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals