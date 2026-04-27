#!/usr/bin/env python3
"""
6h_ElderRay_WeeklyTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation.
Long when Bull Power > 0 AND price > 1w EMA50 AND volume spike.
Short when Bear Power < 0 AND price < 1w EMA50 AND volume spike.
Exit when Elder Power reverses OR loses 1w EMA50 alignment.
Elder Ray measures buying/selling pressure relative to EMA13; works in bull (strong Bull Power) and bear (strong Bear Power) regimes.
Target: 12-30 trades/year (50-120 over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray on 6h (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1w EMA50 (~50*6=300 6h bars), EMA13, volume avg
    start_idx = max(300, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_1w_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Elder Ray alignment with 1w EMA50 trend and volume spike
            # Long: Bull Power > 0 AND price > 1w EMA50 AND volume spike
            # Short: Bear Power < 0 AND price < 1w EMA50 AND volume spike
            long_condition = (bull_val > 0 and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (bear_val < 0 and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when Bear Power >= 0 (selling pressure appears) OR loses 1w EMA50 alignment
            if bear_val >= 0 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when Bull Power <= 0 (buying pressure appears) OR loses 1w EMA50 alignment
            if bull_val <= 0 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_WeeklyTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0