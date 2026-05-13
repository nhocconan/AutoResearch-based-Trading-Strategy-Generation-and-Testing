#%%
#!/usr/bin/env python3
# 12h_WeeklyVWAP_Breakout_1wTrend_Volume
# Hypothesis: Price breaking above/below weekly VWAP with 1-week EMA50 trend filter and volume confirmation captures momentum with controlled trade frequency.
# Works in bull markets via breakouts above VWAP and in bear markets via breakdowns below VWAP.
# Uses 1-week EMA50 to filter trend direction and volume spike for confirmation, reducing false signals.
# Target: 12-37 trades per year per symbol to minimize fee drag.

name = "12h_WeeklyVWAP_Breakout_1wTrend_Volume"
timeframe = "12h"
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

    # ATR for context (not used in signal)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Get 1w data for weekly VWAP calculation and EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly VWAP: cumulative (price * volume) / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_numerator = (typical_price * df_1w['volume']).cumsum()
    vwap_denominator = df_1w['volume'].cumsum()
    vwap = (vwap_numerator / vwap_denominator).values
    
    # Align to 12h timeframe (available after weekly close)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)

    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume filter: >2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above VWAP + 1w EMA50 uptrend + volume spike
            if (close[i] > vwap_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below VWAP + 1w EMA50 downtrend + volume spike
            elif (close[i] < vwap_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below VWAP or volatility drop
            if close[i] < vwap_aligned[i] or volume[i] < vol_avg_20[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above VWAP or volatility drop
            if close[i] > vwap_aligned[i] or volume[i] < vol_avg_20[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#%%