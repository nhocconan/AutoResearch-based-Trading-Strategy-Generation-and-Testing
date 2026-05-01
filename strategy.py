#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme + 1d EMA34 trend + volume spike.
# Williams %R < -80 for long, > -20 for short with 1d EMA34 trend filter and volume > 2x 24-bar MA.
# Uses discrete sizing (0.25) to minimize fee churn. Designed for 12h timeframe to achieve 50-150 trades over 4 years.
# Works in both bull and bear markets via trend filter and extreme readings that capture reversals.

name = "12h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 12h data (need 14 periods)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 2.0 * 24-period average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 14, 24) + 1  # 35
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Williams %R extreme conditions
        wr_oversold = williams_r[i] < -80.0
        wr_overbought = williams_r[i] > -20.0
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold AND uptrend AND volume spike
            if wr_oversold and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND downtrend AND volume spike
            elif wr_overbought and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought (reversal signal)
            if williams_r[i] > -20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold (reversal signal)
            if williams_r[i] < -80.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals