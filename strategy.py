#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: Camarilla pivot breakout at R1/S1 levels with 12h EMA trend filter and volume confirmation.
Long when price breaks above R1 AND 12h EMA(50) uptrend AND volume spike.
Short when price breaks below S1 AND 12h EMA(50) downtrend AND volume spike.
Uses discrete sizing (0.30) to limit fee churn. Target: 75-200 trades over 4 years (19-50/year).
Works in bull (breakouts with trend) and bear (breakdowns with trend) via 12h regime filter.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for regime filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous day
    # Need daily high, low, close - get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Align to 4h timeframe
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_4h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    range_ = prev_high_4h - prev_low_4h
    R1 = prev_close_4h + (range_ * 1.1 / 12)
    S1 = prev_close_4h - (range_ * 1.1 / 12)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 12h, Camarilla calculation (need 2 days), volume MA(20)
    start_idx = max(50, 48, 20) + 1  # 48 for 2 days of 12h data approx
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(R1[i]) or
            np.isnan(S1[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        close_val = close[i]
        regime_long = close_val > ema_50_12h_aligned[i]  # 12h uptrend
        regime_short = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        vol_conf = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 AND 12h uptrend AND volume spike
            long_signal = (close_val > R1[i]) and regime_long and vol_conf
            
            # Short: price breaks below S1 AND 12h downtrend AND volume spike
            short_signal = (close_val < S1[i]) and regime_short and vol_conf
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price breaks below S1 OR 12h trend flips down
            if (close_val < S1[i]) or (not regime_long):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price breaks above R1 OR 12h trend flips up
            if (close_val > R1[i]) or (not regime_short):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0