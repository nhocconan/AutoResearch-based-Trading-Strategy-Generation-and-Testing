#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses price channel breakout for trend capture with 12h EMA50 for regime alignment
# Volume > 2.0x average confirms institutional participation
# Discrete position sizing (0.25) with opposite Donchian(10) exit for symmetry
# Designed for ~20-40 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via trend filter and symmetric exits

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period Donchian channels for entry
    donchian_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period Donchian channels for exit (symmetric)
    donchian_10_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_10_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian20 and 12h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_20_high[i]) or np.isnan(donchian_20_low[i]) or 
            np.isnan(donchian_10_high[i]) or np.isnan(donchian_10_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_20_high = donchian_20_high[i]
        curr_donchian_20_low = donchian_20_low[i]
        curr_donchian_10_high = donchian_10_high[i]
        curr_donchian_10_low = donchian_10_low[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 10-period Donchian low (symmetric exit)
            if curr_close < curr_donchian_10_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 10-period Donchian high (symmetric exit)
            if curr_close > curr_donchian_10_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above 20-period Donchian high, 12h EMA50 up-trend, volume confirmed
            if curr_high > curr_donchian_20_high and curr_close > curr_ema50_12h and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-period Donchian low, 12h EMA50 down-trend, volume confirmed
            elif curr_low < curr_donchian_20_low and curr_close < curr_ema50_12h and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals