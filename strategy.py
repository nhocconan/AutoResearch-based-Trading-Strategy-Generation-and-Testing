#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold) AND 1d EMA34 rising AND volume > 1.5x 20-period MA.
Short when Williams %R crosses below -20 (overbought) AND 1d EMA34 falling AND volume > 1.5x 20-period MA.
Exit when Williams %R crosses below -50 for longs or above -50 for shorts, or EMA trend reverses.
Williams %R captures momentum extremes, 1d EMA34 filters major trend, volume confirms reversal strength.
Works in both bull and bear markets by fading extremes in the direction of higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate 6h Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(14, n):
        # Use lookback of 14 periods (excluding current bar to avoid look-ahead)
        highest_high[i] = np.max(high[i-14:i])
        lowest_low[i] = np.min(low[i-14:i])
        if highest_high[i] != lowest_low[i]:  # Avoid division by zero
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Williams %R (needs 14), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R crossover signals
        williams_cross_above_80 = False
        williams_cross_below_20 = False
        williams_cross_above_50 = False
        williams_cross_below_50 = False
        
        if i >= start_idx + 1:
            williams_prev = williams_r[i-1]
            williams_curr = williams_r[i]
            williams_cross_above_80 = williams_prev <= -80 and williams_curr > -80
            williams_cross_below_20 = williams_prev >= -20 and williams_curr < -20
            williams_cross_above_50 = williams_prev <= -50 and williams_curr > -50
            williams_cross_below_50 = williams_prev >= -50 and williams_curr < -50
        
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 1.5x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND EMA34 rising AND volume filter
            if williams_cross_above_80 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) AND EMA34 falling AND volume filter
            elif williams_cross_below_20 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -50 OR EMA34 starts falling
                if williams_cross_below_50 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -50 OR EMA34 starts rising
                if williams_cross_above_50 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0