#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d EMA34 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend) AND volume > 2x 24-period median.
# Short when Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend) AND volume > 2x 24-period median.
# Williams %R identifies exhaustion points; 1d EMA34 filters for major trend alignment; volume spike confirms conviction.
# Works in both bull and bear markets by fading extremes in the direction of the higher timeframe trend.
# Target: 12-37 trades/year on 6h timeframe.

name = "6h_WilliamsR_Extreme_1dEMA34_Volume_v1"
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
    
    # Calculate 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    williams_r = -100 * (highest_high - close) / hh_ll
    
    # Calculate 24-period volume median for volume confirmation (4x6h = 1d)
    vol_median_24 = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Williams %R and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_median_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2x 24-period volume median
        if vol_median_24[i] <= 0 or np.isnan(vol_median_24[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_24[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) AND uptrend AND volume spike
            if williams_r[i] < -80.0 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend AND volume spike
            elif williams_r[i] > -20.0 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (exit oversold) OR trend turns down
            if williams_r[i] > -50.0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (exit overbought) OR trend turns up
            if williams_r[i] < -50.0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals