#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold) AND 1d EMA34 is rising AND volume > 1.5x 20-period average.
Short when Williams %R crosses below -20 (overbought) AND 1d EMA34 is falling AND volume > 1.5x 20-period average.
Exit when Williams %R crosses below -50 for longs OR above -50 for shorts.
Uses 1d HTF for EMA34 trend and Williams %R to reduce whipsaws. Target: 50-150 total trades over 4 years (12-37/year).
Williams %R: (highest_high - close) / (highest_high - lowest_low) * -100 over 14 periods.
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
    
    # Calculate 1d Williams %R (14-period)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    williams_r = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high == lowest_low:
            williams_r[i] = -50.0
        else:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100.0
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 (34), volume MA (20), Williams %R (14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        wr = williams_r_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Williams %R cross detection
        wr_cross_above_80 = False
        wr_cross_below_20 = False
        wr_cross_above_50 = False
        wr_cross_below_50 = False
        
        if i >= start_idx + 1:
            wr_prev = williams_r_aligned[i-1]
            wr_cross_above_80 = wr_prev <= -80 and wr > -80
            wr_cross_below_20 = wr_prev >= -20 and wr < -20
            wr_cross_above_50 = wr_prev <= -50 and wr > -50
            wr_cross_below_50 = wr_prev >= -50 and wr < -50
        
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
                # Long exit: Williams %R crosses below -50
                if wr_cross_below_50:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -50
                if wr_cross_above_50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_Reversal_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0