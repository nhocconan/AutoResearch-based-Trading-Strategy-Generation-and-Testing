#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Long when price breaks above Donchian(20) high AND weekly close > weekly open (bullish week) AND volume > 2.0x 20-bar average.
# Short when price breaks below Donchian(20) low AND weekly close < weekly open (bearish week) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Weekly pivot filter ensures alignment with higher timeframe momentum, reducing false breakouts.
# Volume spike threshold set to 2.0x to avoid choppy market noise.
# Primary timeframe: 6h, HTF: 1w for weekly bias.

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly bias: 1 = bullish week (close > open), -1 = bearish week (close < open)
    weekly_bias_raw = np.where(df_1w['close'].values > df_1w['open'].values, 1,
                               np.where(df_1w['close'].values < df_1w['open'].values, -1, 0))
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_raw)
    
    # Calculate Donchian(20) channels from 1d data (more stable than 6h for structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian high: max(high, 20) from previous completed day
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    # Donchian low: min(low, 20) from previous completed day
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high_aligned[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low_aligned[i]  # break below Donchian low
        
        # Weekly bias filter
        bullish_week = weekly_bias_aligned[i] == 1
        bearish_week = weekly_bias_aligned[i] == -1
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND bullish week AND volume confirmation
            if (breakout_up and 
                bullish_week and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND bearish week AND volume confirmation
            elif (breakout_down and 
                  bearish_week and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR weekly bias turns bearish
            if (curr_low < donchian_low_aligned[i] or 
                weekly_bias_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR weekly bias turns bullish
            if (curr_high > donchian_high_aligned[i] or 
                weekly_bias_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals