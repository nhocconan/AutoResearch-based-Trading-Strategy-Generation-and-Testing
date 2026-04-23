#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1w EMA34 trend filter and volume spike confirmation.
Long when Williams %R crosses above -80 from oversold AND 1w EMA34 is rising AND volume > 2.0x 24-period average.
Short when Williams %R crosses below -20 from overbought AND 1w EMA34 is falling AND volume > 2.0x 24-period average.
Exit when Williams %R crosses opposite extreme (-20 for long, -80 for short) or EMA34 reverses direction.
Uses 1w HTF for EMA34 trend to avoid whipsaws in ranging markets. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        hh = highest_high[i]
        ll = lowest_low[i]
        if hh != ll:  # avoid division by zero
            williams_r[i] = (hh - close[i]) / (hh - ll) * -100
        else:
            williams_r[i] = -50  # midpoint when range is zero
    
    # 24-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 34, 24)  # Williams %R (14), EMA34 (34), volume MA (24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate Williams %R crossover signals
        wr_cross_above_80 = False
        wr_cross_below_20 = False
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            wr_cross_above_80 = wr_prev <= -80 and wr > -80
            wr_cross_below_20 = wr_prev >= -20 and wr < -20
        
        # Calculate EMA34 slope for trend direction
        ema_rising = False
        ema_falling = False
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        
        if position == 0:
            # Long: Williams %R crosses above -80 from oversold AND EMA34 rising AND volume spike
            if wr_cross_above_80 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought AND EMA34 falling AND volume spike
            elif wr_cross_below_20 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -20 OR EMA34 starts falling
                if wr_cross_below_20 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -80 OR EMA34 starts rising
                if wr_cross_above_80 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1wEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0