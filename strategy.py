#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA50 trend filter + volume confirmation
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short)
# 1d EMA50 ensures we trade with the higher timeframe trend
# Volume confirmation reduces false signals
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend)
# Discrete position sizing 0.25 balances return and drawdown. Target: 75-150 trades over 4 years.

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeConfirm_v1"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) on 6h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero or NaN with neutral value
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r = np.where(np.isnan(williams_r), -50, williams_r)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20, 50) + 1  # 51 (for Williams %R, volume MA, and EMA50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Williams %R extreme levels
        oversold = williams_r[i] < -80  # Oversold condition for long
        overbought = williams_r[i] > -20  # Overbought condition for short
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold AND uptrend AND volume confirmation
            if oversold and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND downtrend AND volume confirmation
            elif overbought and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns above -50 (momentum fading)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns below -50 (momentum fading)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals