#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R(14) crosses above -80 (oversold) AND 1d EMA50 rising AND 6h volume > 1.5x 20-period MA.
Short when Williams %R(14) crosses below -20 (overbought) AND 1d EMA50 falling AND 6h volume > 1.5x 20-period MA.
Exit when Williams %R crosses opposite threshold (-20 for long, -80 for short) or 1d EMA50 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume confirmation for momentum.
Williams %R is effective in ranging/bear markets for mean reversion entries. Target: 50-150 total trades over 4 years.
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
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # avoid division by zero
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 50, 20)  # Williams %R, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R crossover signals
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            wr_cross_above_80 = williams_r[i] > -80 and wr_prev <= -80
            wr_cross_below_20 = williams_r[i] < -20 and wr_prev >= -20
        else:
            wr_cross_above_80 = False
            wr_cross_below_20 = False
        
        # Calculate EMA50 slope for trend direction
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_50_aligned[i] > ema_prev
            ema_falling = ema_50_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) AND EMA50 rising AND volume filter
            if wr_cross_above_80 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought rejection) AND EMA50 falling AND volume filter
            elif wr_cross_below_20 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -20 (overbought) OR EMA50 starts falling
                if wr_cross_below_20 or (i >= start_idx + 1 and ema_50_aligned[i] < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -80 (oversold) OR EMA50 starts rising
                if wr_cross_above_80 or (i >= start_idx + 1 and ema_50_aligned[i] > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0