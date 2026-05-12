#!/usr/bin/env python3
# 6h_VWAP_Deviation_Trend_Follow
# Hypothesis: Price deviations from VWAP with 12h trend filter and volume confirmation capture mean-reversion in ranging markets and trend continuation in breakouts.
# Works in bull markets via trend-following VWAP breaks, in bear via mean reversion to VWAP during consolidation.
# Target: 15-35 trades/year per symbol.

name = "6h_VWAP_Deviation_Trend_Follow"
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

    # Get 12h data for trend filter (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)

    # Calculate VWAP (session-based: daily reset)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    # Reset VWAP at daily boundaries (00:00 UTC)
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    for d in unique_dates:
        mask = (dates == d)
        if np.any(mask):
            first_idx = np.where(mask)[0][0]
            vwap_num[first_idx] = typical_price[first_idx] * volume[first_idx]
            vwap_den[first_idx] = volume[first_idx]
            for i in range(first_idx + 1, len(vwap)):
                if dates[i] != dates[i-1]:
                    vwap_num[i] = typical_price[i] * volume[i]
                    vwap_den[i] = volume[i]
                else:
                    vwap_num[i] = vwap_num[i-1] + typical_price[i] * volume[i]
                    vwap_den[i] = vwap_den[i-1] + volume[i]
            # Recompute VWAP for this day
            for i in range(first_idx, len(vwap)):
                if dates[i] != dates[first_idx]:
                    break
                if vwap_den[i] != 0:
                    vwap[i] = vwap_num[i] / vwap_den[i]
                else:
                    vwap[i] = np.nan

    # VWAP deviation as z-score of recent deviation
    vwap_dev = close - vwap
    vwap_dev_ma = pd.Series(vwap_dev).rolling(window=24, min_periods=24).mean().values  # 24*6h = 6 days
    vwap_dev_std = pd.Series(vwap_dev).rolling(window=24, min_periods=24).std().values
    vwap_z = np.divide(vwap_dev - vwap_dev_ma, vwap_dev_std, out=np.full_like(vwap_dev, np.nan), where=vwap_dev_std!=0)

    # Volume confirmation: volume > 1.5x 24-period average
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        if np.isnan(vwap_z[i]) or np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_avg_24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price below VWAP (mean reversion) + 12h uptrend + volume spike
            if vwap_z[i] < -1.0 and close[i] > ema34_12h_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price above VWAP (mean reversion) + 12h downtrend + volume spike
            elif vwap_z[i] > 1.0 and close[i] < ema34_12h_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above VWAP or 12h trend turns down
            if close[i] > vwap[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below VWAP or 12h trend turns up
            if close[i] < vwap[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals