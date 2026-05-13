# %%
#!/usr/bin/env python3
# 1h_Camarilla_Pivot_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: 1h price breaking above/below Camarilla R1/S1 levels with 4h EMA20 trend filter and 1d volume confirmation.
# Uses 4h EMA20 for trend direction and 1d volume spike for confirmation, reducing false signals.
# Target: 15-37 trades/year per symbol (60-150 total over 4 years) to minimize fee drag.
# Session filter: 08-20 UTC to avoid low-volume periods.
# Position size: 0.20 (discrete levels to reduce churn).

name = "1h_Camarilla_Pivot_R1_S1_Breakout_4hTrend_1dVolume"
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

    # Pre-compute hour filter for session (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)

    # ATR for context (not used in signal)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)

    # Get 1d data for Camarilla pivot calculation and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # 1d volume average (20-day) for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + 4h EMA20 uptrend + 1d volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema20_4h_aligned[i] and
                volume[i] > vol_avg_20_1d_aligned[i] * 2.0):
                signals[i] = 0.20
                position = 1
            # SHORT: Close below S1 + 4h EMA20 downtrend + 1d volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema20_4h_aligned[i] and
                  volume[i] > vol_avg_20_1d_aligned[i] * 2.0):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal
            if close[i] < s1_aligned[i] or close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal
            if close[i] > r1_aligned[i] or close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals
# %%