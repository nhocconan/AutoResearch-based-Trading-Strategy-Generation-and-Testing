#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike
Hypothesis: On 1h timeframe, Camarilla pivot levels (R1/S1) from the previous day capture institutional interest.
Break above R1 with 1d volume spike and 4h uptrend signals long; break below S1 with 1d volume spike and 4h downtrend signals short.
Uses 4h for trend direction, 1d for volume confirmation, and 1h for precise entry timing. Discrete position sizing (0.20) limits trades to 15-37/year.
Session filter (08-20 UTC) reduces noise. Works in both bull/bear markets by trading breakouts with HTF alignment.
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
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d data for Camarilla levels and volume spike (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's Camarilla levels (R1, S1)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d > (2.0 * vol_ma_20_1d))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 4h EMA (20), 1d Camarilla, 1d volume MA (20)
    start_idx = max(20, 0)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with 1d volume spike and 4h uptrend
            long_breakout = (curr_close > camarilla_r1_aligned[i]) and vol_spike_1d_aligned[i] and (curr_close > ema_20_4h_aligned[i])
            # Short: price breaks below Camarilla S1 with 1d volume spike and 4h downtrend
            short_breakout = (curr_close < camarilla_s1_aligned[i]) and vol_spike_1d_aligned[i] and (curr_close < ema_20_4h_aligned[i])
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price breaks below Camarilla S1 OR trend turns down
            if (curr_close < camarilla_s1_aligned[i]) or (curr_close < ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price breaks above Camarilla R1 OR trend turns up
            if (curr_close > camarilla_r1_aligned[i]) or (curr_close > ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0