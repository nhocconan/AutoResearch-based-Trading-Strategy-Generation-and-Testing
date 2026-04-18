#!/usr/bin/env python3
"""
6h_Supertrend_EMA34_VolumeFilter_V1
Hypothesis: Use 1d Supertrend (ATR=10, mult=3) for trend direction, 6h EMA34 for entry timing, and volume > 1.5x 20-period average for confirmation. 
Go long when 6h price crosses above EMA34 AND 1d Supertrend is uptrend, short when price crosses below EMA34 AND 1d Supertrend is downtrend. 
Supertrend adapts to volatility, reducing whipsaws in sideways markets. Volume filter ensures momentum confirmation. 
Target: 15-30 trades/year by combining trend-following with volatility-adjusted entry and volume confirmation. 
Works in bull markets via trend following and in bear via short signals, with reduced false signals during consolidation.
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
    
    # Get 1d data for Supertrend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Supertrend (ATR=10, mult=3)
    atr_period = 10
    multiplier = 3
    
    # Calculate ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full_like(close_1d, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.full_like(close_1d, np.nan)
    uptrend = np.full_like(close_1d, True)
    
    if len(close_1d) >= atr_period:
        supertrend[atr_period-1] = hl2[atr_period-1]
        uptrend[atr_period-1] = True
        
        for i in range(atr_period, len(close_1d)):
            # Update bands
            if close_1d[i-1] > upper_band[i-1]:
                upper_band[i] = hl2[i] + multiplier * atr[i]
            else:
                upper_band[i] = min(upper_band[i-1], hl2[i] + multiplier * atr[i])
            
            if close_1d[i-1] < lower_band[i-1]:
                lower_band[i] = hl2[i] - multiplier * atr[i]
            else:
                lower_band[i] = max(lower_band[i-1], hl2[i] - multiplier * atr[i])
            
            # Determine trend
            if close_1d[i] > upper_band[i-1]:
                uptrend[i] = True
            elif close_1d[i] < lower_band[i-1]:
                uptrend[i] = False
            else:
                uptrend[i] = uptrend[i-1]
                if uptrend[i]:
                    lower_band[i] = max(lower_band[i-1], hl2[i] - multiplier * atr[i])
                else:
                    upper_band[i] = min(upper_band[i-1], hl2[i] + multiplier * atr[i])
            
            supertrend[i] = lower_band[i] if uptrend[i] else upper_band[i]
    
    # Align Supertrend uptrend to 6h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    
    # 6h EMA34
    ema_period = 34
    ema = np.full_like(close, np.nan)
    if len(close) >= ema_period:
        ema_multiplier = 2 / (ema_period + 1)
        ema[ema_period-1] = np.mean(close[:ema_period])
        for i in range(ema_period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * ema_multiplier + ema[i-1]
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, ema_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(uptrend_1d_aligned[i]) or np.isnan(ema[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price crosses above EMA34 AND 1d Supertrend uptrend AND volume confirmation
        if (position <= 0 and 
            close[i] > ema[i] and 
            close[i-1] <= ema[i-1] and 
            uptrend_1d_aligned[i] > 0.5 and 
            volume[i] > 1.5 * vol_ma[i]):
            signals[i] = 0.25
            position = 1
        # Short entry: price crosses below EMA34 AND 1d Supertrend downtrend AND volume confirmation
        elif (position >= 0 and 
              close[i] < ema[i] and 
              close[i-1] >= ema[i-1] and 
              uptrend_1d_aligned[i] < 0.5 and 
              volume[i] > 1.5 * vol_ma[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite crossover
        elif position == 1 and close[i] < ema[i] and close[i-1] >= ema[i-1]:
            signals[i] = -0.25  # reverse to short
            position = -1
        elif position == -1 and close[i] > ema[i] and close[i-1] <= ema[i-1]:
            signals[i] = 0.25   # reverse to long
            position = 1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_Supertrend_EMA34_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0