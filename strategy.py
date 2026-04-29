#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and 1w pivot confirmation
# Donchian breakout captures momentum; 1d EMA50 ensures alignment with daily trend to avoid counter-trend whipsaws
# 1w pivot (weekly high/low) acts as institutional reference: breakouts above weekly high or below weekly low with volume confirmation signal strong moves
# Volume confirmation (>1.5x 20-period average) filters for participation
# Discrete sizing (0.25) minimizes fee churn
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_Donchian20_1dEMA50_1wPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 1 or len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w pivot levels (weekly high and low)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate Donchian channels on 6h timeframe: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 1d EMA50, Donchian(20) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_weekly_high = weekly_high_aligned[i]
        curr_weekly_low = weekly_low_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish (price below 1d EMA50)
            if curr_low < curr_donchian_low or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish (price above 1d EMA50)
            if curr_high > curr_donchian_high or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND above 1d EMA50 AND above weekly high AND volume confirmation
            if (curr_high > curr_donchian_high and 
                curr_close > curr_ema_1d and 
                curr_close > curr_weekly_high and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND below 1d EMA50 AND below weekly low AND volume confirmation
            elif (curr_low < curr_donchian_low and 
                  curr_close < curr_ema_1d and 
                  curr_close < curr_weekly_low and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals