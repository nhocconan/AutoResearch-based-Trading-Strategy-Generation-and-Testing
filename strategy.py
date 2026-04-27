#!/usr/bin/env python3
"""
4h_TRIX_9_VolumeSpike_1dTrend_HTF
Hypothesis: TRIX(9) zero-cross with 1d EMA34 trend filter and volume confirmation. 
TRIX filters noise and catches momentum shifts. In bull markets, long when TRIX>0 and rising; 
in bear markets, short when TRIX<0 and falling. Volume spike confirms conviction. 
1d trend ensures alignment with higher timeframe direction. Designed for 20-40 trades/year 
on 4h to minimize fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(9): triple EMA of close, then percent change
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = (ema3.pct_change() * 100).values  # TRIX as percentage
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for TRIX (3*9=27), EMA34, volume average
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        trix_val = trix[i]
        
        if position == 0:
            # Flat - look for entry: TRIX zero-cross with trend and volume
            # Long: TRIX crosses above zero AND rising AND 1d trend up AND volume spike
            # Short: TRIX crosses below zero AND falling AND 1d trend down AND volume spike
            if i > 0:
                trix_prev = trix[i-1]
                trix_rising = trix_val > trix_prev
                trix_falling = trix_val < trix_prev
            else:
                trix_rising = False
                trix_falling = False
            
            long_condition = (trix_val > 0 and trix_prev <= 0 and trix_rising and 
                            close_val > ema_trend and vol_spike)
            short_condition = (trix_val < 0 and trix_prev >= 0 and trix_falling and 
                             close_val < ema_trend and vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when TRIX turns negative OR 1d trend turns down
            if trix_val <= 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TRIX turns positive OR 1d trend turns up
            if trix_val >= 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_9_VolumeSpike_1dTrend_HTF"
timeframe = "4h"
leverage = 1.0