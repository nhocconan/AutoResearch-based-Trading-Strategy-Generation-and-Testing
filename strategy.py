#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h trend filter and volume confirmation.
# Williams %R < -80 = oversold (long setup), > -20 = overbought (short setup).
# Enter only when price crosses back above/below -50 level with 12h EMA50 trend alignment.
# Volume spike (>1.5x 20-period average) confirms momentum. Works in bull (buy dips in uptrend) 
# and bear (sell rallies in downtrend). Discrete sizing 0.25 balances return and drawdown.
# Target: 50-150 total trades over 4 years (12-37/year).

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
    
    # Williams %R (14-period) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 14, 20) + 1  # 51 (for EMA50, Williams %R, and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
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
        
        # Williams %R extreme levels and crossover logic
        williams_r_prev = williams_r[i-1]
        williams_r_curr = williams_r[i]
        
        # Long setup: Williams %R was oversold (< -80) and crosses above -50
        long_setup = (williams_r_prev < -80) and (williams_r_curr > -50)
        # Short setup: Williams %R was overbought (> -20) and crosses below -50
        short_setup = (williams_r_prev > -20) and (williams_r_curr < -50)
        
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
            # Exit on Williams %R overbought (> -20) or trend change
            if williams_r_curr > -20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold (< -80) or trend change
            if williams_r_curr < -80 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals