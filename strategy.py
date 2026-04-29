#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d HTF for EMA trend filter and 20-period Donchian channels from 1d data
# Volume > 1.8x average confirms breakout strength
# Discrete position sizing (0.25) with Donchian(10) exit for quick profit taking
# Designed for ~12-25 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via 1d EMA34 trend filter - only trades in direction of higher timeframe trend

name = "12h_Donchian20_1dEMA34_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA trend filter and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: highest high of last 20 periods
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 1d Donchian exit channels (10-period) for profit taking
    upper_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    lower_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Align exit channels to 12h timeframe
    upper_10_aligned = align_htf_to_ltf(prices, df_1d, upper_10)
    lower_10_aligned = align_htf_to_ltf(prices, df_1d, lower_10)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume MA and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(upper_10_aligned[i]) or 
            np.isnan(lower_10_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_upper_20 = upper_20_aligned[i]
        curr_lower_20 = lower_20_aligned[i]
        curr_upper_10 = upper_10_aligned[i]
        curr_lower_10 = lower_10_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 10-period Donchian lower band (profit taking)
            if curr_close < curr_lower_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 10-period Donchian upper band (profit taking)
            if curr_close > curr_upper_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above 20-period upper channel with 1d EMA34 uptrend and volume confirmation
            if curr_high > curr_upper_20 and curr_close > curr_ema34_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-period lower channel with 1d EMA34 downtrend and volume confirmation
            elif curr_low < curr_lower_20 and curr_close < curr_ema34_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals