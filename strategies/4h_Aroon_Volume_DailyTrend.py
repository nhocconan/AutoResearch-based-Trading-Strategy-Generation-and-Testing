#!/usr/bin/env python3
# 4h Aroon Oscillator + Volume Spike + Daily Trend Filter
# Hypothesis: Aroon Oscillator identifies strong trends (values near +100/-100 indicate strong uptrend/downtrend).
# Combined with volume spikes to confirm institutional participation and daily EMA trend filter,
# this strategy captures strong momentum moves while avoiding chop. Designed for low trade frequency (~20-30/year).
# Works in both bull and bear markets by following the trend as defined by higher timeframe.

name = "4h_Aroon_Volume_DailyTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Aroon Oscillator (period=25) ===
    # Days since highest high
    highest_high_idx = pd.Series(high).rolling(window=25, min_periods=1).apply(lambda x: x.argmax(), raw=True)
    days_since_high = 24 - highest_high_idx  # 25-period: 0 to 24 days ago
    
    # Days since lowest low
    lowest_low_idx = pd.Series(low).rolling(window=25, min_periods=1).apply(lambda x: x.argmin(), raw=True)
    days_since_low = 24 - lowest_low_idx
    
    # Aroon Up and Down
    aroon_up = ((25 - days_since_high) / 25) * 100
    aroon_down = ((25 - days_since_low) / 25) * 100
    
    # Aroon Oscillator
    aroon_osc = aroon_up - aroon_down  # Range: -100 to +100
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(aroon_osc[i]) or np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Aroon Oscillator > 50 (strong uptrend) + volume spike + price above daily EMA34
            if (aroon_osc[i] > 50 and 
                vol_spike[i] and
                close[i] > ema_34_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Aroon Oscillator < -50 (strong downtrend) + volume spike + price below daily EMA34
            elif (aroon_osc[i] < -50 and 
                  vol_spike[i] and
                  close[i] < ema_34_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Trend weakens (Aroon Oscillator < 0)
            if aroon_osc[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens (Aroon Oscillator > 0)
            if aroon_osc[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals