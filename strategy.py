#!/usr/bin/env python3
"""
12h Williams %R Reversal + 1d EMA50 Trend + Volume Confirmation
Hypothesis: Williams %R(14) identifies overbought/oversold conditions on 12h chart.
In ranging/bear markets (2025+), extreme %R readings often precede mean-reversion reversals.
Filtered by 1d EMA50 trend (avoid counter-trend trades) and volume confirmation (institutional interest).
Designed for lower trade frequency (~15-25/year) to minimize fee drag while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 12h chart
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Readings: 0 to -20 = overbought, -80 to -100 = oversold
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    williams_r = ((highest_high - close) / hl_range) * -100
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Williams %R, EMA50_1d, and volume MA to propagate
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        williams_r_val = williams_r[i]
        ema50_1d = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: Williams %R crosses above -80 from oversold AND uptrend (price > 1d EMA50) AND volume confirmation
            long_condition = (williams_r_val > -80) and (williams_r_val < -20) and (curr_close > ema50_1d) and volume_confirm
            # Short: Williams %R crosses below -20 from overbought AND downtrend (price < 1d EMA50) AND volume confirmation
            short_condition = (williams_r_val < -20) and (williams_r_val > -80) and (curr_close < ema50_1d) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) or crosses above -20 (overbought)
            if williams_r_val < -50 or williams_r_val > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss) or crosses below -80 (oversold)
            if williams_r_val > -50 or williams_r_val < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Reversal_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0