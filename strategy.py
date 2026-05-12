#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Trade breakouts from Camarilla R1/S1 levels on 4h with 12h EMA50 trend filter and volume confirmation.
Camarilla levels provide precise support/resistance in ranging markets, while 12h EMA filters for higher-timeframe trend direction.
Volume spike confirms institutional participation. Designed for 20-50 trades/year on 4h timeframe.
Works in bull/bear by following 12h trend, avoids false breakouts via volume confirmation.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical = (high + low + close) / 3
    range_ = high - low
    # Camarilla levels
    R4 = close + range_ * 1.500
    R3 = close + range_ * 1.250
    R2 = close + range_ * 1.166
    R1 = close + range_ * 1.083
    S1 = close - range_ * 1.083
    S2 = close - range_ * 1.166
    S3 = close - range_ * 1.250
    S4 = close - range_ * 1.500
    return R1, S1, R2, S2, R3, S3, R4, S4, typical

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for EMA50 trend ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # Calculate 12h EMA50
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.roll(ema_50, 1)
    ema_50_prev[0] = ema_50[0]
    
    # Align 12h EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_50_prev_aligned = align_htf_to_ltf(prices, df_12h, ema_50_prev)

    # Calculate Camarilla levels from previous 1d (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    R1, S1, _, _, _, _, _, _, _ = calculate_camarilla(prev_high, prev_low, prev_close)
    
    # Align Camarilla levels to 4h (using same 1d data)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # 4h volume spike: current > 2.0x average of last 6 periods (24h)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_50_prev_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 + 12h uptrend + volume spike
            if (close[i] > R1_aligned[i] and 
                ema_50_aligned[i] > ema_50_prev_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + 12h downtrend + volume spike
            elif (close[i] < S1_aligned[i] and 
                  ema_50_aligned[i] < ema_50_prev_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or 12h trend turns down
            if (close[i] < S1_aligned[i] or 
                ema_50_aligned[i] < ema_50_prev_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or 12h trend turns up
            if (close[i] > R1_aligned[i] or 
                ema_50_aligned[i] > ema_50_prev_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals