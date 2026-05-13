#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Enter long when price breaks above Camarilla R1 level during 4h uptrend with volume confirmation, short when breaks below S1 during 4h downtrend.
# Uses Camarilla pivot levels (H4/L4 and R1/S1) from 4h for structure, 1h for entry timing.
# Volume filter ensures institutional participation. Trend filter from 4h EMA50 reduces false signals in chop.
# Designed for low frequency: Camarilla levels are strong S/R, breakouts require momentum.
# Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend).

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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

    # Get 4h data for Camarilla pivot and trend
    df_4h = get_htf_data(prices, '4h')
    
    # Typical price for Camarilla calculation
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    tp_h = typical_price.values
    tp_l = df_4h['low'].values
    tp_c = df_4h['close'].values
    
    # Calculate Camarilla levels (based on previous day's OHLC)
    # Using 4h bar's high/low/close as session equivalent
    H = tp_h
    L = tp_l
    C = tp_c
    
    # Camarilla formulas
    R4 = C + ((H - L) * 1.1 / 2)
    R3 = C + ((H - L) * 1.1 / 4)
    R2 = C + ((H - L) * 1.1 / 6)
    R1 = C + ((H - L) * 1.1 / 12)
    S1 = C - ((H - L) * 1.1 / 12)
    S2 = C - ((H - L) * 1.1 / 6)
    S3 = C - ((H - L) * 1.1 / 4)
    S4 = C - ((H - L) * 1.1 / 2)
    
    # 4h trend: EMA50
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume spike: volume > 1.5 * 12-period average (equivalent to 6h)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > 1.5 * vol_ma_12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + 4h uptrend + volume spike
            if close[i] > R1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Close < S1 + 4h downtrend + volume spike
            elif close[i] < S1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA50 or below S1 (stop)
            if close[i] < ema50_4h_aligned[i] or close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close above EMA50 or above R1 (stop)
            if close[i] > ema50_4h_aligned[i] or close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals