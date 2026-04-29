#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses Donchian channel breakouts for trend capture with 1d EMA34 as strong trend filter
# Volume > 1.8x average confirms institutional participation and reduces false breakouts
# Discrete position sizing (0.25) with Donchian(10) exit for quick profit taking
# Designed for ~20-40 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via trend filter - only trades in direction of 1d EMA34

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v2"
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
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    # Donchian upper = max(high, lookback)
    # Donchian lower = min(low, lookback)
    lookback_entry = 20
    lookback_exit = 10
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_upper_20 = high_series.rolling(window=lookback_entry, min_periods=lookback_entry).max().values
    donchian_lower_20 = low_series.rolling(window=lookback_entry, min_periods=lookback_entry).min().values
    donchian_upper_10 = high_series.rolling(window=lookback_exit, min_periods=lookback_exit).max().values
    donchian_lower_10 = low_series.rolling(window=lookback_exit, min_periods=lookback_exit).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_entry, 34, 20)  # Donchian20, 1d EMA34, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_20[i]) or np.isnan(donchian_lower_20[i]) or
            np.isnan(donchian_upper_10[i]) or np.isnan(donchian_lower_10[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_dc_upper_20 = donchian_upper_20[i]
        curr_dc_lower_20 = donchian_lower_20[i]
        curr_dc_upper_10 = donchian_upper_10[i]
        curr_dc_lower_10 = donchian_lower_10[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Donchian lower(10) (profit taking or reversal)
            if curr_close < curr_dc_lower_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Donchian upper(10) (profit taking or reversal)
            if curr_close > curr_dc_upper_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 1.8x 20-period average
            vol_spike = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above Donchian upper(20) with 1d EMA34 uptrend and volume spike
            if curr_close > curr_dc_upper_20 and curr_close > curr_ema34_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower(20) with 1d EMA34 downtrend and volume spike
            elif curr_close < curr_dc_lower_20 and curr_close < curr_ema34_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals