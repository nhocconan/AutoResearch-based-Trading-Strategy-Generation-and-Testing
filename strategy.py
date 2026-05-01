#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h EMA50 trend filter and volume confirmation.
# Williams %R < -80 = oversold (long setup), > -20 = overbought (short setup).
# Enter on %R crossing back above -80 (long) or below -20 (short) with 12h trend and volume spike.
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend).
# Discrete position sizing 0.25 targets ~100 trades over 4 years (25/year) to avoid fee drag.

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
    # Lookback period 14
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 50, 20) + 1
    
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
        
        # Williams %R conditions
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Long setup: %R crosses above -80 from below (exiting oversold)
        long_setup = (wr > -80) and (wr_prev <= -80)
        # Short setup: %R crosses below -20 from above (exiting overbought)
        short_setup = (wr < -20) and (wr_prev >= -20)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R long setup AND uptrend AND volume confirmation
            if long_setup and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R short setup AND downtrend AND volume confirmation
            elif short_setup and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R crossing below -50 (momentum loss) or short setup
            if wr < -50 or short_setup:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R crossing above -50 (momentum loss) or long setup
            if wr > -50 or long_setup:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals