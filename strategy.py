#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_VolumeConfirm
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
Goes long when price breaks above R1 with 12h uptrend and volume spike.
Short when price breaks below S1 with 12h downtrend and volume spike.
Exit when price returns to Camarilla pivot point (PP) or trend reverses.
Uses discrete sizing (0.25) to minimize fees. Target: 20-50 trades/year.
Works in bull via breakouts with trend, in bear via mean reversion at pivot levels.
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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla pivot levels (based on previous 4h bar)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_4h = (high_4h + low_4h + close_4h) / 3
    r1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12
    s1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (already aligned via get_htf_data)
    # Since we're using 4h data on 4h timeframe, no alignment needed
    pp_4h_aligned = pp_4h
    r1_4h_aligned = r1_4h
    s1_4h_aligned = s1_4h
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(pp_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1, 12h uptrend, volume spike
            long_signal = (close[i] > r1_4h_aligned[i]) and (close[i] > ema_50_12h_aligned[i]) and vol_spike[i]
            # Short: price breaks below S1, 12h downtrend, volume spike
            short_signal = (close[i] < s1_4h_aligned[i]) and (close[i] < ema_50_12h_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to PP or trend turns bearish
            exit_signal = (close[i] <= pp_4h_aligned[i]) or (close[i] < ema_50_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to PP or trend turns bullish
            exit_signal = (close[i] >= pp_4h_aligned[i]) or (close[i] > ema_50_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0