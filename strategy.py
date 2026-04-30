#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels
# (>80 for short, <20 for long) with 12h trend alignment and volume spike provide
# high-probability mean-reversion entries in both bull and bear markets.
# Low trade frequency target: 12-30/year to minimize fee drag on 6h timeframe.

name = "6h_WilliamsR_Reversal_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R(14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Trend filter: price relative to 12h EMA50
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (<20) reversing up in uptrend with volume
            if is_uptrend and curr_wr < -20 and williams_r[i-1] >= -20 and curr_volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (>80) reversing down in downtrend with volume
            elif is_downtrend and curr_wr > -80 and williams_r[i-1] <= -80 and curr_volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit on Williams %R overbought or trend change
            if curr_wr > -20 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit on Williams %R oversold or trend change
            if curr_wr < -80 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals