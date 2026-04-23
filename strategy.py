#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold) AND 1d EMA34 is rising AND volume > 1.5x 20-period average.
Short when Williams %R crosses below -20 (overbought) AND 1d EMA34 is falling AND volume > 1.5x 20-period average.
Exit when Williams %R crosses -50 (mean reversion) or trend reverses.
Uses 1d HTF for EMA34 trend to reduce whipsaws in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
Williams %R formula: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if rr[i] > 0:
            williams_r[i] = (highest_high[i] - close[i]) / rr[i] * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 20)  # Williams %R (14), EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate Williams %R crossovers
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_cross_above_80 = wr_prev <= -80 and wr > -80
        wr_cross_below_20 = wr_prev >= -20 and wr < -20
        wr_cross_above_50 = wr_prev <= -50 and wr > -50
        wr_cross_below_50 = wr_prev >= -50 and wr < -50
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND EMA34 rising AND volume spike
            if wr_cross_above_80 and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) AND EMA34 falling AND volume spike
            elif wr_cross_below_20 and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50 OR EMA34 starts falling
                if wr_cross_above_50 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50 OR EMA34 starts rising
                if wr_cross_below_50 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dEMA34_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0