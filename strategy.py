#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeConfirm_v1
Hypothesis: On 1d timeframe, Camarilla R1/S1 breakouts with 1w EMA34 trend filter and volume confirmation (>1.5x 20-day average) provides high-probability directional signals with controlled trade frequency. The weekly EMA34 captures the primary trend, reducing counter-trend trades. Long when price > 1w EMA34 + breaks above R1 + volume spike; short when price < 1w EMA34 + breaks below S1 + volume spike. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 30-100 trades over 4 years (7-25/year) for optimal 1d frequency. Works in bull markets (trend following) and bear markets (trend following with short signals). Volume confirmation ensures institutional participation, reducing false signals in low-volume environments.
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
    
    # Get 1d data for HTF indicators (Camarilla pivots from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # need at least 2 days for previous day's OHLC
        return np.zeros(n)
    
    # Get 1w data for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d OHLC for Camarilla pivot levels (previous 1d bar)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed)
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
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
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
            # Long: price > 1w EMA34 + breaks above R1 + volume
            long_signal = (close[i] > ema_34_1w_aligned[i] and 
                          close[i] > camarilla_r1_aligned[i] and 
                          vol_confirmed)
            
            # Short: price < 1w EMA34 + breaks below S1 + volume
            short_signal = (close[i] < ema_34_1w_aligned[i] and 
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
            # Exit: price closes below 1w EMA34 OR breaks below S1 (reversal)
            if close[i] < ema_34_1w_aligned[i] or close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above 1w EMA34 OR breaks above R1 (reversal)
            if close[i] > ema_34_1w_aligned[i] or close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0