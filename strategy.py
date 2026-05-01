#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 12h EMA50 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions. Extreme readings (<-90 or >-10) 
# combined with 12h EMA50 trend filter and volume confirmation provide high-probability 
# mean-reversion entries in ranging markets and trend continuation in strong moves.
# Discrete position sizing 0.25 to limit drawdown. Target: 20-50 trades/year.
# Works in bull/bear via trend alignment - only takes longs in uptrend, shorts in downtrend.

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R on 6h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Lookback period: 14
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 50, 20) + 1  # 51 (for EMA50, Williams %R, and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
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
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Williams %R extreme levels
        williams_r_oversold = williams_r[i] < -90  # Extremely oversold
        williams_r_overbought = williams_r[i] > -10  # Extremely overbought
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R extremely oversold AND uptrend AND volume confirmation
            if williams_r_oversold and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R extremely overbought AND downtrend AND volume confirmation
            elif williams_r_overbought and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns above -50 (momentum fading) OR stoploss via reverse signal
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns below -50 (momentum fading) OR stoploss via reverse signal
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals