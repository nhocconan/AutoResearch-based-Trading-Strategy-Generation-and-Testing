#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel breakouts for trend capture with 1w EMA50 as strong trend filter
# Volume > 2.0x average confirms institutional participation
# Discrete position sizing (0.25) with opposite Donchian breakout exit
# Designed for ~12-25 trades/year on 12h timeframe to minimize fee drag
# Works in bull/bear via 1w trend filter - only trades in direction of weekly trend

name = "12h_Donchian20_1wEMA50_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period Donchian channels on 1d timeframe for structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian high/low on daily
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 30-period average volume for confirmation on 12h timeframe
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Volume and 1w EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_dc_high = donchian_high_aligned[i]
        curr_dc_low = donchian_low_aligned[i]
        curr_vol_ma = vol_ma_30[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low (trend reversal)
            if curr_low < curr_dc_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (trend reversal)
            if curr_high > curr_dc_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 30-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above Donchian high, 1w EMA50 up-trend, volume confirmed
            if curr_high > curr_dc_high and curr_close > curr_ema50_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low, 1w EMA50 down-trend, volume confirmed
            elif curr_low < curr_dc_low and curr_close < curr_ema50_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals