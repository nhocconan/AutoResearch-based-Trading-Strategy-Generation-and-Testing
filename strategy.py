#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold) AND 1d EMA34 is rising AND volume > 1.5x 20-period average.
Short when Williams %R crosses below -20 (overbought) AND 1d EMA34 is falling AND volume > 1.5x 20-period average.
Exit when Williams %R crosses below -50 for longs or above -50 for shorts, or EMA trend reverses.
Williams %R = (highest high - close) / (highest high - lowest low) * -100 over 14 periods.
Uses 1d HTF for EMA34 trend to reduce whipsaws in ranging markets. Target: 50-150 total trades over 4 years (12-37/year).
Williams %R is effective in capturing mean reversals in both bull and bear markets when combined with trend filter.
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
    
    # Calculate Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.full_like(close, np.nan, dtype=np.float64)
    mask = rr != 0
    williams_r[mask] = ((highest_high[mask] - close[mask]) / rr[mask]) * -100
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 20, 34)  # Williams %R (14), volume MA (20), EMA34 (34)
    
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
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND EMA34 rising AND volume spike
            if i >= start_idx + 1:
                wr_prev = williams_r[i-1]
                wr_cross_up = wr_prev <= -80 and wr > -80
            else:
                wr_cross_up = False
            
            if wr_cross_up and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND EMA34 falling AND volume spike
            elif i >= start_idx + 1:
                wr_prev = williams_r[i-1]
                wr_cross_down = wr_prev >= -20 and wr < -20
            else:
                wr_cross_down = False
                
            if wr_cross_down and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -50 OR EMA34 starts falling
                if i >= start_idx + 1:
                    wr_prev = williams_r[i-1]
                    wr_cross_down_mid = wr_prev >= -50 and wr < -50
                else:
                    wr_cross_down_mid = False
                    
                if wr_cross_down_mid or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -50 OR EMA34 starts rising
                if i >= start_idx + 1:
                    wr_prev = williams_r[i-1]
                    wr_cross_up_mid = wr_prev <= -50 and wr > -50
                else:
                    wr_cross_up_mid = False
                    
                if wr_cross_up_mid or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
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