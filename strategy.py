#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and 1d volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power increasing (less negative) AND price > 1d EMA34 AND volume spike
# Short when Bear Power < 0 AND Bull Power decreasing (less positive) AND price < 1d EMA34 AND volume spike
# Uses 1d EMA34 for trend alignment and volume spike for institutional participation
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Works in bull/bear: trend filter ensures we trade with higher timeframe momentum

name = "6h_ElderRay_BullBear_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume spike: volume > 2.0 * 20-period EMA
    vol_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    volume_spike_1d = vol_1d > (2.0 * vol_ema_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Rate of change of Bear Power (to detect improving bear power = less negative)
    bear_power_change = np.diff(bear_power, prepend=bear_power[0])
    # Rate of change of Bull Power (to detect weakening bull power = less positive)
    bull_power_change = np.diff(bull_power, prepend=bull_power[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA13 and 1d indicators
    start_idx = max(14, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power positive AND Bear Power improving (increasing) AND price > 1d EMA34 AND volume spike
            if (bull_power[i] > 0 and 
                bear_power_change[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power negative AND Bull Power weakening (decreasing) AND price < 1d EMA34 AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power_change[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative OR Bear Power accelerates downward OR price < 1d EMA34
            if (bull_power[i] <= 0 or 
                bear_power_change[i] < -0.1 or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR Bull Power accelerates upward OR price > 1d EMA34
            if (bear_power[i] >= 0 or 
                bull_power_change[i] > 0.1 or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals