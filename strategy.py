#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly pivot levels (standard calculation)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point = (H + L + C)/3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Resistance 1 = (2*P) - L
    r1_1w = (2 * pp_1w) - low_1w
    # Support 1 = (2*P) - H
    s1_1w = (2 * pp_1w) - high_1w
    
    # Align weekly pivot to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation: current volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 periods = 6 days for 6h
    volume_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Price relative to weekly pivot levels
        above_r1 = close[i] > r1_1w_aligned[i]
        below_s1 = close[i] < s1_1w_aligned[i]
        between_pivots = (close[i] >= s1_1w_aligned[i]) and (close[i] <= r1_1w_aligned[i])
        
        # Entry conditions
        # Long: price above R1 in uptrend with volume confirmation (breakout)
        long_entry = above_r1 and uptrend and volume_confirm[i]
        # Short: price below S1 in downtrend with volume confirmation (breakdown)
        short_entry = below_s1 and downtrend and volume_confirm[i]
        
        # Exit conditions: return to pivot area or trend reversal
        if position == 1:
            exit_condition = (not uptrend) or between_pivots
        elif position == -1:
            exit_condition = (not downtrend) or between_pivots
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0