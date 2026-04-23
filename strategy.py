#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 from below (oversold bounce) AND 1d EMA34 is rising AND volume > 2.0x 20-period average.
Short when Williams %R crosses below -20 from above (overbought rejection) AND 1d EMA34 is falling AND volume > 2.0x 20-period average.
Exit when Williams %R crosses opposite extreme (-20 for long, -80 for short) or EMA34 trend reverses.
Uses 1d HTF for EMA34 trend to reduce whipsaws. Target: 50-150 total trades over 4 years (12-37/year).
Williams %R(14) = (Highest High - Close) / (Highest High - Lowest Low) * -100.
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
    
    # Calculate Williams %R(14) on 6h data
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
            williams_r[i] = -50  # neutral when no range
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 34)  # Williams %R (14), volume MA (20), EMA34 (34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Williams %R crossover signals
        wr_cross_above_80 = False  # crossed above -80 (from below)
        wr_cross_below_20 = False  # crossed below -20 (from above)
        
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            # Bullish crossover: wr crosses above -80 (from below -80 to above -80)
            if wr_prev <= -80 and wr > -80:
                wr_cross_above_80 = True
            # Bearish crossover: wr crosses below -20 (from above -20 to below -20)
            if wr_prev >= -20 and wr < -20:
                wr_cross_below_20 = True
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) AND EMA34 rising AND volume spike
            if wr_cross_above_80 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought rejection) AND EMA34 falling AND volume spike
            elif wr_cross_below_20 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -20 (overbought) OR EMA34 starts falling
                if wr_cross_below_20 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -80 (oversold) OR EMA34 starts rising
                if wr_cross_above_80 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Extreme_1dEMA34_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0