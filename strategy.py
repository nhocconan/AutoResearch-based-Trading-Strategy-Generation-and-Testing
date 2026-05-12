#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout with Volume Spike and 1d Trend Filter
Hypothesis: Camarilla pivot levels act as strong support/resistance. A breakout above R1 or below S1
with volume confirmation and aligned with the daily trend captures institutional moves.
Works in bull markets via breakouts and in bear markets via breakdowns, with volume filtering false signals.
"""
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # === 1d TREND (EMA 34) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Camarilla levels from previous day ===
    # Use previous day's OHLC to avoid look-ahead
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = df_1d['close'].values[0]  # first value
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    # Camarilla R1, S1 calculation
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_1d[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend + price breaks above R1 + volume spike
            if (close[i] > trend_1d[i] and 
                close[i] > r1_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price breaks below S1 + volume spike
            elif (close[i] < trend_1d[i] and 
                  close[i] < s1_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend changes
            if close[i] < s1_aligned[i] or close[i] < trend_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend changes
            if close[i] > r1_aligned[i] or close[i] > trend_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals