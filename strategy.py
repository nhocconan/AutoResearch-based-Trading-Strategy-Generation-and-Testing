#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter_VolumeSpike_v1
Hypothesis: Elder Ray (Bull Power/Bear Power) on 6h with 1d EMA trend filter and volume spike confirmation. 
- Bull Power = High - EMA(13) measures bullish strength
- Bear Power = EMA(13) - Low measures bearish strength
- Long when Bull Power > 0 AND Bear Power rising (momentum) AND price > 1d EMA50 (uptrend) AND volume spike
- Short when Bear Power > 0 AND Bull Power falling (momentum) AND price < 1d EMA50 (downtrend) AND volume spike
- Exit when power diverges or volume drops
- Works in bull markets (buying strength) and bear markets (selling strength). Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray on 6h: EMA13 of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Power momentum (change from previous bar)
    bull_power_mom = bull_power - np.roll(bull_power, 1)
    bear_power_mom = bear_power - np.roll(bear_power, 1)
    bull_power_mom[0] = 0
    bear_power_mom[0] = 0
    
    # Volume spike: current volume > 1.8 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup
    start_idx = max(30, 13, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for new directional momentum with volume confirmation
            # Long: Bull Power positive AND rising AND price above 1d EMA50 (uptrend) AND volume spike
            long_entry = (bull_power[i] > 0) and (bull_power_mom[i] > 0) and \
                         (close_val > ema_50_1d_aligned[i]) and volume_spike[i]
            # Short: Bear Power positive AND rising AND price below 1d EMA50 (downtrend) AND volume spike
            short_entry = (bear_power[i] > 0) and (bear_power_mom[i] > 0) and \
                          (close_val < ema_50_1d_aligned[i]) and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when bull power fails or turns down
            exit_condition = (bull_power[i] <= 0) or (bull_power_mom[i] < 0) or \
                           (close_val < ema_50_1d_aligned[i])  # trend break
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when bear power fails or turns down
            exit_condition = (bear_power[i] <= 0) or (bear_power_mom[i] < 0) or \
                           (close_val > ema_50_1d_aligned[i])  # trend break
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0