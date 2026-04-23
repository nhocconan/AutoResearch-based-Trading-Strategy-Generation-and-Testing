#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 12h EMA34 trend filter and 1d volume spike confirmation.
Long when Williams %R(14) crosses above -80 (oversold) AND 12h EMA34 is rising AND 1d volume > 1.8x 20-period average.
Short when Williams %R(14) crosses below -20 (overbought) AND 12h EMA34 is falling AND 1d volume > 1.8x 20-period average.
Exit when Williams %R crosses below -50 for long or above -50 for short, or 12h EMA34 reverses direction.
Williams %R captures mean reversion in overextended moves, while 12h EMA34 ensures we trade with the higher timeframe trend.
Volume spike filters for institutional participation. Target: 80-180 total trades over 4 years (20-45/year) for 6h timeframe.
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
    
    # Calculate 12h EMA34 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 1d Williams %R(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):  # Start from index 13 for 14-period lookback
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high == lowest_low:
            williams_r[i] = -50  # Avoid division by zero
        else:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d volume average for spike filter (HTF)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 13, 20)  # EMA34 (34), Williams %R (13), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        wr = williams_r_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
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
            if i >= start_idx + 1:
                wr_prev = williams_r_aligned[i-1]
                wr_cross_up = wr_prev <= -80 and wr > -80
            else:
                wr_cross_up = False
            
            if wr_cross_up and ema_rising and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) AND EMA34 falling AND volume spike
            elif i >= start_idx + 1:
                wr_prev = williams_r_aligned[i-1]
                wr_cross_down = wr_prev >= -20 and wr < -20
            else:
                wr_cross_down = False
            
            if wr_cross_down and ema_falling and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -50 OR EMA34 starts falling
                if i >= start_idx + 1:
                    wr_prev = williams_r_aligned[i-1]
                    wr_cross_down_50 = wr_prev >= -50 and wr < -50
                else:
                    wr_cross_down_50 = False
                
                if wr_cross_down_50 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -50 OR EMA34 starts rising
                if i >= start_idx + 1:
                    wr_prev = williams_r_aligned[i-1]
                    wr_cross_up_50 = wr_prev <= -50 and wr > -50
                else:
                    wr_cross_up_50 = False
                
                if wr_cross_up_50 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_12hEMA34_Trend_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0