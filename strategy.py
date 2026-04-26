#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume spike (>2x avg) provides high-conviction entries with low frequency. Long when price > 1d EMA34 + breaks above R1 + volume spike. Short when price < 1d EMA34 + breaks below S1 + volume spike. Discrete sizing (0.0, ±0.25) minimizes fee churn. Targets 75-150 total trades over 4 years (19-37/year) to avoid fee drag. Works in bull/bear via 1d trend filter aligning with higher timeframe momentum.
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
    
    # Get 1d data for HTF trend filter and Camarilla pivots (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d OHLC for Camarilla pivot levels (previous day)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average
        
        if position == 0:
            # Long: price > 1d EMA34 + breaks above R1 + volume
            long_signal = (close[i] > ema_34_1d_aligned[i] and 
                          close[i] > camarilla_r1_aligned[i] and 
                          vol_confirmed)
            
            # Short: price < 1d EMA34 + breaks below S1 + volume
            short_signal = (close[i] < ema_34_1d_aligned[i] and 
                           close[i] < camarilla_s1_aligned[i] and 
                           vol_confirmed)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below 1d EMA34 OR breaks below S1 (reversal)
            if close[i] < ema_34_1d_aligned[i] or close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above 1d EMA34 OR breaks above R1 (reversal)
            if close[i] > ema_34_1d_aligned[i] or close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0