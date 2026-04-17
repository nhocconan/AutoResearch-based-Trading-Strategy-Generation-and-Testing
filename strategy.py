#!/usr/bin/env python3
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
    
    # === 12h Williams %R (14-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full_like(high_12h, np.nan)
    lowest_low = np.full_like(low_12h, np.nan)
    period = 14
    for i in range(len(high_12h)):
        if i >= period - 1:
            highest_high[i] = np.max(high_12h[i-(period-1):i+1])
            lowest_low[i] = np.min(low_12h[i-(period-1):i+1])
        elif i > 0:
            highest_high[i] = np.max(high_12h[0:i+1])
            lowest_low[i] = np.min(low_12h[0:i+1])
        else:
            highest_high[i] = high_12h[0]
            lowest_low[i] = low_12h[0]
    
    williams_r = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_12h[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # === 12h Williams %R smoothing (3-period) to reduce noise ===
    williams_r_smooth = np.full_like(williams_r, np.nan)
    for i in range(len(williams_r)):
        if i >= 2:
            williams_r_smooth[i] = np.mean(williams_r[i-2:i+1])
        elif i > 0:
            williams_r_smooth[i] = np.mean(williams_r[0:i+1])
        else:
            williams_r_smooth[i] = williams_r[0]
    
    # === 12h EMA(34) for trend filter ===
    ema_34 = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema_34[33] = np.mean(close_12h[:34])  # seed
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema_34[i] = alpha * close_12h[i] + (1 - alpha) * ema_34[i-1]
    else:
        for i in range(len(close_12h)):
            ema_34[i] = np.mean(close_12h[:i+1]) if i >= 0 else close_12h[0]
    
    # === Align indicators to 4h timeframe ===
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r_smooth)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # === 4h Volume confirmation ===
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    # === Williams %R levels ===
    OVERBOUGHT = -20
    OVERSOLD = -80
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND price above EMA34
            if (williams_r_aligned[i] > OVERSOLD and 
                williams_r_aligned[i-1] <= OVERSOLD and  # crossed up
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Williams %R crosses below -20 from above AND price below EMA34
            elif (williams_r_aligned[i] < OVERBOUGHT and 
                  williams_r_aligned[i-1] >= OVERBOUGHT and  # crossed down
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Williams %R crosses below -50 OR crosses above -20
            if (williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50) or \
               (williams_r_aligned[i] < OVERBOUGHT and williams_r_aligned[i-1] >= OVERBOUGHT):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -50 OR crosses below -80
            if (williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50) or \
               (williams_r_aligned[i] > OVERSOLD and williams_r_aligned[i-1] <= OVERSOLD):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_EMA34_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0