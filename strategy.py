#!/usr/bin/env python3
"""
1d_Keltner_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Keltner Channel (ATR-based) breakouts on 1d with 1w EMA50 trend filter and volume confirmation. 
Keltner Channels adapt to volatility better than fixed % bands, reducing false breakouts in ranging markets. 
The 1w EMA50 provides strong long-term trend alignment that works in both bull and bear markets. 
Volume confirmation ensures breakouts occur with institutional participation. 
Targeting 30-80 total trades over 4 years (7-20/year) to minimize fee drag while capturing major moves.
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
    
    # Load 1d data ONCE before loop for Keltner calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(10) on 1d for Keltner Channel
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr1.iloc[0] = 0
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Keltner Channel (20 EMA middle, ATR*2 bands)
    ema_20 = df_1d['close'].ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + (2.0 * atr_10)
    lower_keltner = ema_20 - (2.0 * atr_10)
    
    # Align Keltner levels to 1d timeframe (no additional delay needed for EMA/ATR)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = df_1w['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection: volume > 2.0 * 20-period average volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 50, 20, 10, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_keltner_aligned[i]) or
            np.isnan(lower_keltner_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or
            np.isnan(atr_10[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Long logic: price breaks above upper Keltner with volume spike + in uptrend
        if close[i] > upper_keltner_aligned[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below lower Keltner with volume spike + in downtrend
        elif close[i] < lower_keltner_aligned[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to middle line or trend weakens
        elif position == 1 and (close[i] < ema_20_aligned[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > ema_20_aligned[i] or not downtrend):
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

name = "1d_Keltner_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0