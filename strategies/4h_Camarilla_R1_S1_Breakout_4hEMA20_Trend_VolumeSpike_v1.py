#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_4hEMA20_Trend_VolumeSpike_v1
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 4h EMA20 trend filter and volume spike (>1.5x avg) provides high-probability directional signals with controlled trade frequency. R1/S1 levels represent intraday support/resistance, reducing false breakouts. Long when price > 4h EMA20 + breaks above R1 + volume spike; short when price < 4h EMA20 + breaks below R1 + volume spike. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 75-150 trades over 4 years (19-37/year) for optimal 4h frequency. Works in bull markets (trend following) and bear markets (trend following with short signals). Volume confirmation ensures institutional participation, reducing false signals in low-volume environments.
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
    
    # Get 4h data for HTF indicators (EMA20, Camarilla pivots)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # need enough for EMA20
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 4h OHLC for Camarilla pivot levels (previous 4h bar)
    o_4h = df_4h['open'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_4h + (h_4h - l_4h) * 1.1 / 12
    camarilla_s1 = c_4h - (h_4h - l_4h) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_confirmed = vol_ratio[i] > 1.5  # volume at least 1.5x average
        
        if position == 0:
            # Long: price > 4h EMA20 + breaks above R1 + volume
            long_signal = (close[i] > ema_20_4h_aligned[i] and 
                          close[i] > camarilla_r1_aligned[i] and 
                          vol_confirmed)
            
            # Short: price < 4h EMA20 + breaks below S1 + volume
            short_signal = (close[i] < ema_20_4h_aligned[i] and 
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
            # Exit: price closes below 4h EMA20 OR breaks below S1 (reversal)
            if close[i] < ema_20_4h_aligned[i] or close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above 4h EMA20 OR breaks above R1 (reversal)
            if close[i] > ema_20_4h_aligned[i] or close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_4hEMA20_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0