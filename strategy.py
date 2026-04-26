#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: On 1d timeframe, price breaks above Camarilla R1 or below S1 with 1w EMA34 trend filter and volume confirmation capture institutional moves in both bull and bear markets. Camarilla levels derived from prior day's range provide accurate support/resistance. Volume spike confirms participation. 1w EMA34 ensures alignment with major trend. Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # Load 1w data ONCE before loop for HTF trend filter (EMA)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # Use prior day's data to avoid look-ahead
    prior_high = np.concatenate([[high[0]], high[:-1]])
    prior_low = np.concatenate([[low[0]], low[:-1]])
    prior_close = np.concatenate([[close[0]], close[:-1]])
    
    camarilla_range = prior_high - prior_low
    r1 = prior_close + 1.1 * camarilla_range / 12.0
    s1 = prior_close - 1.1 * camarilla_range / 12.0
    
    # Volume spike detection (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1[i]) or
            np.isnan(s1[i]) or
            np.isnan(volume_ema[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA34)
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Long logic: price breaks above R1 with volume spike + in uptrend or ranging (not strong downtrend)
        if close[i] > r1[i] and volume_spike[i] and not downtrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below S1 with volume spike + in downtrend or ranging (not strong uptrend)
        elif close[i] < s1[i] and volume_spike[i] and not uptrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to prior close or trend reverses strongly
        elif position == 1 and (close[i] < prior_close[i] or downtrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > prior_close[i] or uptrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0