#!/usr/bin/env python3
"""
6h_RSI_2_Timeframe_Confluence
Hypothesis: Combining RSI(2) mean reversion on 6h with weekly trend filter (EMA200) and volume confirmation creates high-probability mean-reversion trades. RSI(2) captures extreme short-term reversals, while weekly EMA200 ensures alignment with longer-term trend. Volume spike confirms institutional participation. Works in both bull and bear markets by following the weekly trend direction.
"""

name = "6h_RSI_2_Timeframe_Confluence"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Weekly EMA200 trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)

    # RSI(2) on 6h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[1] = np.mean(gain[:2]) if len(gain) >= 2 else gain[0] if len(gain) >= 1 else 0
    avg_loss[1] = np.mean(loss[:2]) if len(loss) >= 2 else loss[0] if len(loss) >= 1 else 0
    
    for i in range(2, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_2 = 100 - (100 / (1 + rs))

    # Volume spike: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(rsi_2[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI(2) oversold + price above weekly EMA200 + volume spike
            if (rsi_2[i] < 15 and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI(2) overbought + price below weekly EMA200 + volume spike
            elif (rsi_2[i] > 85 and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI(2) overbought or price crosses below weekly EMA200
            if (rsi_2[i] > 70 or close[i] < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI(2) oversold or price crosses above weekly EMA200
            if (rsi_2[i] < 30 or close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals