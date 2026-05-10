#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend
# Hypothesis: Camarilla pivot levels on 1h provide precise entry/exit points while 4h trend (EMA50) filters direction.
# Long when price breaks above R1 in 4h uptrend, short when breaks below S1 in 4h downtrend.
# Volume confirmation reduces false breakouts. Designed for low trade frequency (15-37/year) and works in both bull/bear markets by following 4h trend.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels for 1h using previous 4h bar's OHLC
    # Camarilla uses previous period's range: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use the completed 4h bar's OHLC to calculate levels for the current 1h session
    ph = df_4h['high'].values  # previous 4h high
    pl = df_4h['low'].values   # previous 4h low
    pc = df_4h['close'].values # previous 4h close
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume filter: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 4h
        uptrend = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]  # rising EMA = uptrend
        downtrend = ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]  # falling EMA = downtrend
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma[i] * 1.5  # 50% above average volume
        
        if position == 0:
            # Long entry: price breaks above R1 in 4h uptrend with volume
            if close[i] > r1_aligned[i] and uptrend and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 in 4h downtrend with volume
            elif close[i] < s1_aligned[i] and downtrend and vol_ok:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns down
            if close[i] < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns up
            if close[i] > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals