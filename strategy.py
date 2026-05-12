#!/usr/bin/env python3
# 4h_Keltner_Channel_Breakout_1wTrend_Volume
# Hypothesis: Keltner Channel breakouts with 1-week EMA trend filter and volume spikes
# capture strong momentum moves. Works in bull markets via upper band breaks and
# in bear markets via lower band breaks. The 1-week trend filter ensures we only
# trade in the direction of the higher timeframe trend, reducing whipsaws.
# Volume spikes confirm institutional interest. Low trade frequency expected due
# to confluence of three conditions: channel break, trend alignment, and volume.

name = "4h_Keltner_Channel_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1w Data for EMA Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Keltner Channel (20, 2) ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(ema20[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close above upper Keltner + price above 1w EMA50 + volume spike
            if close[i] > upper_keltner[i] and close[i] > ema_50_4h[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below lower Keltner + price below 1w EMA50 + volume spike
            elif close[i] < lower_keltner[i] and close[i] < ema_50_4h[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close below EMA20 (middle of Keltner) or trend change
            if close[i] < ema20[i] or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA20 (middle of Keltner) or trend change
            if close[i] > ema20[i] or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals