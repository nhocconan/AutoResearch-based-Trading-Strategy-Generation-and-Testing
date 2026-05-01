#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d trend filter + volume spike
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# In strong trends, extreme readings can precede continuations (not reversals)
# Entry: Williams %R < -90 (deep oversold) in 1d uptrend OR > -10 (deep overbought) in 1d downtrend
# Volume spike confirms participation (> 2x 24-period average)
# Works in bull (buy deep oversold with uptrend) and bear (sell deep overbought with downtrend)
# Discrete position sizing 0.25 balances return and drawdown
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_WilliamsR_Extreme_1dTrend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HH - C) / (HH - LL)
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: current 6h volume > 2.0 * 24-period average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 24) + 1  # 35 (for EMA34 and volume MA24)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Williams %R extreme conditions
        williams_extreme_oversold = williams_r_aligned[i] < -90  # Deep oversold
        williams_extreme_overbought = williams_r_aligned[i] > -10  # Deep overbought
        
        if position == 0:  # Flat - look for new entries
            # Long: Deep oversold AND uptrend AND volume spike
            if williams_extreme_oversold and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Deep overbought AND downtrend AND volume spike
            elif williams_extreme_overbought and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns above -50 (exit oversold zone)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns below -50 (exit overbought zone)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals