#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly Bollinger Band breakout + weekly ADX trend filter
# Long when price breaks above upper BB(20,2) on weekly AND weekly ADX > 25 (trending)
# Short when price breaks below lower BB(20,2) on weekly AND weekly ADX > 25
# Exit when price returns to weekly BB middle (20-period SMA) or weekly ADX < 20
# Uses weekly timeframe for structure to capture major trends and avoid whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for Bollinger Bands and ADX
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_weekly).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_weekly).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20
    
    # Calculate weekly ADX (14-period)
    # True Range
    tr1 = high_weekly[1:] - low_weekly[1:]
    tr2 = np.abs(high_weekly[1:] - close_weekly[:-1])
    tr3 = np.abs(low_weekly[1:] - close_weekly[:-1])
    tr_first = np.max([high_weekly[0] - low_weekly[0], 
                       np.abs(high_weekly[0] - close_weekly[0]), 
                       np.abs(low_weekly[0] - close_weekly[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_weekly[1:] - high_weekly[:-1]) > (low_weekly[:-1] - low_weekly[1:]), 
                       np.maximum(high_weekly[1:] - high_weekly[:-1], 0), 0)
    dm_minus = np.where((low_weekly[:-1] - low_weekly[1:]) > (high_weekly[1:] - high_weekly[:-1]), 
                        np.maximum(low_weekly[:-1] - low_weekly[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly indicators to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_weekly, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_weekly, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_weekly, middle_bb)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price breaks BB + ADX filter
        breakout_up = close[i] > upper_bb_aligned[i]
        breakout_down = close[i] < lower_bb_aligned[i]
        strong_trend = adx_aligned[i] > 25
        
        long_entry = breakout_up and strong_trend
        short_entry = breakout_down and strong_trend
        
        # Exit conditions: price returns to middle BB OR ADX weakens
        exit_long = position == 1 and (close[i] < middle_bb_aligned[i] or adx_aligned[i] < 20)
        exit_short = position == -1 and (close[i] > middle_bb_aligned[i] or adx_aligned[i] < 20)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_bb_adx_breakout"
timeframe = "6h"
leverage = 1.0