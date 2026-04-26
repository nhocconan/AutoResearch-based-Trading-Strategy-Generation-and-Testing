#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, Camarilla R4/S4 breakouts with 1d EMA34 trend filter and volume confirmation (>2x avg) provides high-probability directional signals. R4/S4 represent stronger breakout levels than R3/S3, reducing false signals. Works in bull markets (long when price > 1d EMA34 + R4 breakout) and bear markets (short when price < 1d EMA34 + S4 breakdown). Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 trades over 4 years (12-37/year) for optimal 6h frequency. 1d trend filter avoids whipsaws while volume spike confirms institutional participation.
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
    
    # Get daily data for HTF trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d OHLC for Camarilla pivot levels (previous day)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    camarilla_r4 = c_1d + (h_1d - l_1d) * 1.1 / 2
    camarilla_s4 = c_1d - (h_1d - l_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ratio[i])):
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
            # Long: price > daily EMA34 + breaks above R4 + volume
            long_signal = (close[i] > ema_34_1d_aligned[i] and 
                          close[i] > camarilla_r4_aligned[i] and 
                          vol_confirmed)
            
            # Short: price < daily EMA34 + breaks below S4 + volume
            short_signal = (close[i] < ema_34_1d_aligned[i] and 
                           close[i] < camarilla_s4_aligned[i] and 
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
            # Exit: price closes below daily EMA34 OR breaks below S4 (reversal)
            if close[i] < ema_34_1d_aligned[i] or close[i] < camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above daily EMA34 OR breaks above R4 (reversal)
            if close[i] > ema_34_1d_aligned[i] or close[i] > camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0