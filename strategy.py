#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
# Long when %R crosses above -80 from below in uptrend, short when crosses below -20 from above in downtrend.
# Williams %R identifies overbought/oversold conditions; combined with trend filter captures reversals in both bull and bear markets.
# Discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "12h_WilliamsR_Reversal_1dEMA50_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 14-period Williams %R on 12h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Williams %R (14), EMA50 (50), and volume median (20)
    start_idx = max(14, 50, 20) + 1  # 51
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.8)
        
        # Williams %R reversal signals
        # Long: %R crosses above -80 from below (oversold reversal)
        williams_r_long_signal = (curr_williams_r > -80) and (prev_williams_r <= -80)
        # Short: %R crosses below -20 from above (overbought reversal)
        williams_r_short_signal = (curr_williams_r < -20) and (prev_williams_r >= -20)
        
        if position == 0:  # Flat - look for new entries
            # Long: Oversold reversal AND uptrend AND volume confirmation
            if williams_r_long_signal and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Overbought reversal AND downtrend AND volume confirmation
            elif williams_r_short_signal and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when %R crosses above -20 (overbought) or trend changes
            if curr_williams_r > -20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when %R crosses below -80 (oversold) or trend changes
            if curr_williams_r < -80 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals