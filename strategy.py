#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wEMA20_Trend_VolumeS
Hypothesis: Price breaking above/below Camarilla R1/S1 levels on 12h with 1w EMA20 trend filter and volume confirmation (1.5x average) captures strong trending moves while avoiding false breakouts. Camarilla levels provide precise intraday support/resistance, 1w EMA20 ensures alignment with weekly trend, and volume filter adds confirmation. Target: 20-40 trades/year per symbol. Works in bull/bear by following weekly trend direction.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wEMA20_Trend_VolumeS"
timeframe = "12h"
leverage = 1.0

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
    
    # Camarilla levels for 12h (based on previous day's range)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # But for intraday, we use previous bar's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # HTF: 1w EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + 1w EMA20 uptrend + volume spike
            if (close[i] > r1[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 1w EMA20 downtrend + volume spike
            elif (close[i] < s1[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (support)
            if close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (resistance)
            if close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals