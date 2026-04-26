#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeConfirm_v1
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA20 trend filter and 1d volume spike confirmation.
The strategy uses higher timeframes for signal direction (4h trend, 1d volume regime) and 1h only for precise entry timing.
Camarilla pivots provide mathematically derived support/resistance levels that work well in crypto.
Targeting 80-120 total trades over 4 years (20-30/year) to balance signal quality and fee drag.
Works in both bull and bear markets by using EMA trend filter and volume confirmation to avoid false breakouts.
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
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Camarilla levels on 1h data (using typical price)
    typical_price = (high + low + close) / 3.0
    # Use 5-period lookback for pivot calculation (standard for intraday)
    lookback = 5
    # Rolling high, low, close for pivot calculation
    roll_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    roll_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    roll_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Camarilla equations: Pivot = (H+L+C)/3
    pivot = (roll_high + roll_low + roll_close) / 3.0
    # R1 = C + (H-L)*1.1/12
    r1 = roll_close + (roll_high - roll_low) * 1.1 / 12.0
    # S1 = C - (H-L)*1.1/12
    s1 = roll_close - (roll_high - roll_low) * 1.1 / 12.0
    
    # Volume spike detection on 1d: volume > 1.5 * 20-period average
    volume_spike = volume > (1.5 * vol_ma_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(lookback, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(pivot[i]) or
            np.isnan(r1[i]) or
            np.isnan(s1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend filter (EMA20)
        uptrend = close[i] > ema_20_aligned[i]
        downtrend = close[i] < ema_20_aligned[i]
        
        # Long logic: price breaks above R1 with volume spike + in uptrend
        if close[i] > r1[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short logic: price breaks below S1 with volume spike + in downtrend
        elif close[i] < s1[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: price returns to opposite level or trend weakens
        elif position == 1 and (close[i] < s1[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r1[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0