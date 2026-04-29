#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(15) breakout with 1d EMA34 trend filter and volume confirmation
# Uses tight Donchian channels for high-probability breakouts in trending markets
# 1d EMA34 provides strong trend filter to avoid counter-trend trades
# Volume > 1.3x average confirms institutional participation and reduces false breakouts
# Discrete position sizing (0.25) with Donchian(8) exit for quick profit taking
# Designed for ~15-30 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via trend filter - only trades in direction of 1d EMA34

name = "12h_Donchian15_1dEMA34_VolumeConfirm_v1"
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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian channels (15-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=15, min_periods=15).max().values
    donchian_lower = low_series.rolling(window=15, min_periods=15).min().values
    
    # Calculate 12h Donchian channels (8-period) for exit
    donchian_upper_8 = high_series.rolling(window=8, min_periods=8).max().values
    donchian_lower_8 = low_series.rolling(window=8, min_periods=8).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15, 34)  # Donchian and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_upper_8[i]) or np.isnan(donchian_lower_8[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_donchian_upper_8 = donchian_upper_8[i]
        curr_donchian_lower_8 = donchian_lower_8[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 8-period Donchian lower (quick profit taking)
            if curr_close < curr_donchian_lower_8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 8-period Donchian upper (quick profit taking)
            if curr_close > curr_donchian_upper_8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.3x 20-period average
            vol_confirmed = curr_volume > 1.3 * curr_vol_ma
            
            # Long when price breaks above 15-period Donchian upper, 1d EMA34 up-trend, volume confirmed
            if curr_high > curr_donchian_upper and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 15-period Donchian lower, 1d EMA34 down-trend, volume confirmed
            elif curr_low < curr_donchian_lower and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals