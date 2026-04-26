#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v6
Hypothesis: On 12h timeframe, trade Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume spike confirmation. Designed for low frequency (12-37 trades/year) to minimize fee drag. Works in both bull and bear markets via 1d trend filter and avoids low-volume false breakouts. Uses discrete position sizing (0.25) to reduce churn.
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
    
    # Get 1d data for HTF trend (EMA34) and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(34) 1d, volume MA (20)
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        if position == 0:
            # Long: break above Camarilla R1 with uptrend and volume spike
            long_signal = (high_val > camarilla_r1_val) and \
                          uptrend and \
                          vol_spike
            
            # Short: break below Camarilla S1 with downtrend and volume spike
            short_signal = (low_val < camarilla_s1_val) and \
                           downtrend and \
                           vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend reversal
            if close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal
            if close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v6"
timeframe = "12h"
leverage = 1.0