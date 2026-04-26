#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirm
Hypothesis: Camarilla R4/S4 breakout with 1w trend filter (price vs weekly EMA34) and volume spike confirmation.
Long when price breaks above R4 with volume > 1.5x MA20 and close > weekly EMA34.
Short when price breaks below S4 with volume > 1.5x MA20 and close < weekly EMA34.
Uses discrete sizing (0.25) to minimize fee drag. Target: 50-150 trades over 4 years.
Works in bull/bear via 1w trend filter and Camarilla extreme levels as strong support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 6h (based on previous bar's range)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, S4 = close - 1.1*(high-low)*1.1/2
    # Actually: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    R4 = prev_close + rang * 1.1 / 2
    S4 = prev_close - rang * 1.1 / 2
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Load 1w data for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for volume MA, 34 for EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R4[i]) or np.isnan(S4[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R4 + volume spike + price > weekly EMA34
        long_condition = (close[i] > R4[i]) and volume_spike[i] and (close[i] > ema_34_1w_aligned[i])
        
        # Short logic: break below S4 + volume spike + price < weekly EMA34
        short_condition = (close[i] < S4[i]) and volume_spike[i] and (close[i] < ema_34_1w_aligned[i])
        
        # Exit logic: opposite break or loss of trend
        exit_long = (close[i] < S4[i]) or (close[i] < ema_34_1w_aligned[i])
        exit_short = (close[i] > R4[i]) or (close[i] > ema_34_1w_aligned[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0