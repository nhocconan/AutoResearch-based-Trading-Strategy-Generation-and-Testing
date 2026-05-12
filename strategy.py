# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Volume_Weighted_CCI_Trend
Hypothesis: Combines CCI momentum with volume-weighted price action on 6h timeframe.
Uses 12h trend filter (EMA50) to align with higher timeframe direction and volume confirmation
to filter false signals. Designed for 15-35 trades/year to minimize fee drift while capturing
trends in both bull and bear markets. Volume weighting reduces noise and improves signal quality.
"""

name = "6h_Volume_Weighted_CCI_Trend"
timeframe = "6h"
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

    # Get 12h data (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values

    # Calculate EMA50 on 12h close
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate VWAP on 6h (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(typical_price, np.nan), where=vwap_den!=0)

    # Calculate CCI(20) on 6h typical price
    tp = typical_price
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = np.divide((tp - sma_tp), (0.015 * mad), out=np.full_like(tp, np.nan), where=(mad != 0))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(cci[i]) or np.isnan(vwap[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine 12h trend
        trend_up = close_12h[i // 2] > ema50_12h[i // 2] if i // 2 < len(close_12h) else False
        trend_down = close_12h[i // 2] < ema50_12h[i // 2] if i // 2 < len(close_12h) else False

        if position == 0:
            # LONG: CCI > 50 + price > VWAP + 12h uptrend
            if cci[i] > 50 and close[i] > vwap[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: CCI < -50 + price < VWAP + 12h downtrend
            elif cci[i] < -50 and close[i] < vwap[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CCI < -50 or trend reversal
            if cci[i] < -50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CCI > 50 or trend reversal
            if cci[i] > 50 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals