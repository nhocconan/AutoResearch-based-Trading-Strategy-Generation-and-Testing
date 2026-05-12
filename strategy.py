# USAGE: python3 strategy.py
#!/usr/bin/env python3
# 12h_1D_Camarilla_R1S1_Breakout_R1S1_Reversal
# Hypothesis: Breakouts at daily Camarilla R1/S1 levels with reversal entries at opposite levels.
# In uptrends, buy R1 breakouts; in downtrends, sell S1 breakdowns.
# Reversal trades: sell at R1 in uptrend, buy at S1 in downtrend.
# Uses 1d trend filter (EMA50) and volume confirmation to reduce false signals.
# Designed for 12h to limit trade frequency (target: 50-150 total trades over 4 years).

name = "12h_1D_Camarilla_R1S1_Breakout_R1S1_Reversal"
timeframe = "12h"
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

    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Camarilla levels from previous 1d OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12

    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R1 in uptrend with volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and uptrend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S1 in downtrend with volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and downtrend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla S1 (mean reversion) or trend breaks down
            if close[i] < camarilla_s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla R1 (mean reversion) or trend breaks up
            if close[i] > camarilla_r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals