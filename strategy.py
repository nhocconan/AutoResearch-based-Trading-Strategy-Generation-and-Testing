#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Reversal with 1w EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold reversal) AND 1w EMA34 rising AND volume > 1.3x 20-period MA.
Short when Williams %R crosses below -20 (overbought reversal) AND 1w EMA34 falling AND volume > 1.3x 20-period MA.
Exit when Williams %R crosses below -50 (for long) or above -50 (for short) OR 1w EMA34 reverses.
Uses 1w HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams %R provides timely reversal signals, 1w EMA34 filters major trend, volume confirms reversal strength.
Designed to work in both bull and bear markets by following the higher timeframe trend and capturing mean reversions within the trend.
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
    
    # Calculate 12h Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    for i in range(14, n):
        highest_high = np.max(high[i-14:i+1])
        lowest_low = np.min(low[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # Avoid division by zero
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 12h volume MA (20-period) for spike filter
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
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Williams %R crossovers
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_cross_above_80 = wr_prev <= -80 and wr > -80
        wr_cross_below_20 = wr_prev >= -20 and wr < -20
        wr_cross_below_50 = wr_prev > -50 and wr <= -50
        wr_cross_above_50 = wr_prev < -50 and wr >= -50
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.3x 20-period MA (balanced threshold)
        vol_filter = volume[i] > 1.3 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) AND EMA34 rising AND volume filter
            if wr_cross_above_80 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) AND EMA34 falling AND volume filter
            elif wr_cross_below_20 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -50 OR EMA34 starts falling
                if wr_cross_below_50 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -50 OR EMA34 starts rising
                if wr_cross_above_50 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_Reversal_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0