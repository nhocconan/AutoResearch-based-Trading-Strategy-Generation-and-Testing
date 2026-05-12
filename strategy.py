#!/usr/bin/env python3
"""
1h_PriceAction_4hTrend_1dVolatility
Hypothesis: Trade pullbacks in 1h timeframe aligned with 4h EMA trend and 1d volatility filter.
- Long when: price pulls back to 4h EMA20 in uptrend, 1d ATR < 50-day ATR median (low volatility)
- Short when: price rallies to 4h EMA20 in downtrend, 1d ATR < 50-day ATR median
- Uses session filter (08-20 UTC) to avoid low-liquidity hours
- Target: 15-35 trades/year by requiring trend alignment + volatility filter + session
- Works in bull/bear by following 4h trend; volatility filter avoids choppy markets
"""

name = "1h_PriceAction_4hTrend_1dVolatility"
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

    # Pre-compute session hours (08-20 UTC) to avoid low-liquidity hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    # Get 4h data for EMA20 trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)

    # Get 1d data for ATR volatility filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate True Range and ATR(14) for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period: use high-low
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # 50-day median ATR for volatility regime filter
    atr_median_50 = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).median().values
    atr_median_50_aligned = align_htf_to_ltf(prices, df_1d, atr_median_50)

    # Current 14-period ATR (aligned to 1h)
    tr_l = np.abs(high - low)
    tr_hc = np.abs(high - np.roll(close, 1))
    tr_lc = np.abs(low - np.roll(close, 1))
    tr_1h = np.maximum(tr_l, np.maximum(tr_hc, tr_lc))
    tr_1h[0] = tr_l[0]
    atr_14_1h = pd.Series(tr_1h).ewm(span=14, adjust=False, min_periods=14).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after warmup
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(atr_median_50_aligned[i]) or
            np.isnan(atr_14_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volatility filter: only trade when current 1h ATR < 50% of 1d median ATR (low volatility regime)
        vol_filter = atr_14_1h[i] < (0.5 * atr_median_50_aligned[i])

        if position == 0:
            # LONG: price near 4h EMA20 (within 0.5*ATR) + uptrend + low volatility
            ema_distance = abs(close[i] - ema_20_4h_aligned[i])
            uptrend = close[i] > ema_20_4h_aligned[i]
            if (ema_distance < (0.5 * atr_14_1h[i]) and 
                uptrend and 
                vol_filter):
                signals[i] = 0.20
                position = 1
            # SHORT: price near 4h EMA20 (within 0.5*ATR) + downtrend + low volatility
            elif (ema_distance < (0.5 * atr_14_1h[i]) and 
                  not uptrend and 
                  vol_filter):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price moves away from EMA or trend reverses
            ema_distance = abs(close[i] - ema_20_4h_aligned[i])
            uptrend = close[i] > ema_20_4h_aligned[i]
            if (ema_distance > (1.0 * atr_14_1h[i]) or 
                not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price moves away from EMA or trend reverses
            ema_distance = abs(close[i] - ema_20_4h_aligned[i])
            uptrend = close[i] > ema_20_4h_aligned[i]
            if (ema_distance > (1.0 * atr_14_1h[i]) or 
                uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals