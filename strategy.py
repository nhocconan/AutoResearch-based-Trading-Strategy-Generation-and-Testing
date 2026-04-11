# -*- coding: utf-8 -*-
# 4h_12h_camarilla_breakout_volume_v1
# Strategy: 4h Camarilla pivot breakout with 12h trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as strong S/R. Breakouts aligned with 12h trend and volume
# capture significant moves. Fewer trades (~20-40/year) reduce fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    rng = prev_high - prev_low
    H3 = prev_close + 1.1 * rng / 4
    L3 = prev_close - 1.1 * rng / 4
    H4 = prev_close + 1.1 * rng / 2
    L4 = prev_close - 1.1 * rng / 2
    
    # Align to 4h
    H3_4h = align_htf_to_ltf(prices, df_12h, H3)
    L3_4h = align_htf_to_ltf(prices, df_12h, L3)
    H4_4h = align_htf_to_ltf(prices, df_12h, H4)
    L4_4h = align_htf_to_ltf(prices, df_12h, L4)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 20-period volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or np.isnan(H4_4h[i]) or np.isnan(L4_4h[i]) or
            np.isnan(ema_50_12h_4h[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        breakout_up = high[i] > H3_4h[i-1]
        breakdown_down = low[i] < L3_4h[i-1]
        trend_bullish = close[i] > ema_50_12h_4h[i]
        trend_bearish = close[i] < ema_50_12h_4h[i]
        
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and low[i] < L4_4h[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > H4_4h[i-1]:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals