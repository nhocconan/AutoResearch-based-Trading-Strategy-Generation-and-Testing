#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 1w EMA34 as strong trend filter to avoid counter-trend trades
# Donchian(20) breakout captures strong directional moves
# Volume > 1.8x average confirms institutional participation
# Discrete position sizing (0.25) with opposite Donchian(10) exit for risk control
# Designed for ~10-20 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via 1w EMA34 trend filter - only trades in direction of weekly trend

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) for entry
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian channels (10-period) for exit
    donchian_upper_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_lower_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Donchian(20) and 1w EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_upper_10[i]) or np.isnan(donchian_lower_10[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_donchian_upper_10 = donchian_upper_10[i]
        curr_donchian_lower_10 = donchian_lower_10[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Donchian Lower(10) - stop loss/profit protection
            if curr_close < curr_donchian_lower_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Donchian Upper(10) - stop loss/profit protection
            if curr_close > curr_donchian_upper_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 1.8x 20-period average
            vol_spike = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above Donchian Upper(20) with 1w EMA34 uptrend and volume spike
            if curr_high > curr_donchian_upper and curr_close > curr_ema34_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian Lower(20) with 1w EMA34 downtrend and volume spike
            elif curr_low < curr_donchian_lower and curr_close < curr_ema34_1w and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals