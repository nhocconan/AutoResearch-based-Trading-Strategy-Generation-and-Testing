#!/usr/bin/env python3
"""
6h Williams %R + 12h EMA Trend + Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions on 6h timeframe.
Entries taken when %R crosses above/below key levels (80/20) with 12h EMA trend alignment
and volume confirmation. Works in both bull/bear markets by fading extremes in ranging
markets and catching momentum in trending markets. Targets 75-150 trades over 4 years
(19-37/year) to balance opportunity with fee drag.
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
    
    # Get 12h data for EMA trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h close (only needs completed 12h candle)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period volume MA for 12h volume confirmation
    vol_ma_20_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_20_12h[i] = np.mean(df_12h['volume'].values[i-19:i+1])
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate Williams %R on 6h (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(13, n):  # 14 periods needed (0-13)
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        hh = highest_high[i]
        ll = lowest_low[i]
        if hh != ll:  # avoid division by zero
            williams_r[i] = (hh - close[i]) / (hh - ll) * -100
        else:
            williams_r[i] = -50  # midpoint if range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Williams %R and volume MA
    start_idx = max(20, 13)  # 20 for 12h volume MA, 13 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_12h_aligned[i]
        wr = williams_r[i]
        vol_ma_12h = vol_ma_20_12h_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.8 * 20-period 12h average
        volume_confirm = curr_volume > 1.8 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: Williams %R crosses above 20 from below (oversold), above 12h EMA, volume confirmation
            long_entry = (wr > 20 and 
                         wr_prev <= 20 and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: Williams %R crosses below 80 from above (overbought), below 12h EMA, volume confirmation
            short_entry = (wr < 80 and 
                          wr_prev >= 80 and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Williams %R rises above 80 (overbought) OR price falls below 12h EMA
            if wr > 80 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Williams %R falls below 20 (oversold) OR price rises above 12h EMA
            if wr < 20 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        # Store current Williams %R for next iteration's crossover detection
        wr_prev = wr
    
    return signals

name = "6h_WilliamsR_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0