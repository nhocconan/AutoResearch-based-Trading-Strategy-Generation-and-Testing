#!/usr/bin/env python3
# 4h_WeeklyVWAP_Pullback_RSI_Trend
# Hypothesis: Buy pullbacks to weekly VWAP in bullish weekly trend (price > weekly VWAP) and sell rallies to weekly VWAP in bearish weekly trend (price < weekly VWAP), using 4h RSI for entry timing and volume confirmation. Works in bull (buys dips in uptrend) and bear (sells rallies in downtrend). Target: 20-50 trades/year.

name = "4h_WeeklyVWAP_Pullback_RSI_Trend"
timeframe = "4h"
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

    # Get weekly data for VWAP and trend
    df_1w = get_htf_data(prices, '1w')
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap = vwap.values
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)

    # Weekly trend: price above/below VWAP
    weekly_trend = df_1w['close'] > vwap  # True for bullish, False for bearish
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))

    # 4h RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_vwap_aligned[i]) or np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price near weekly VWAP (pullback) in bullish weekly trend + RSI oversold + volume
            if (close[i] <= weekly_vwap_aligned[i] * 1.005 and  # within 0.5% above VWAP
                weekly_trend_aligned[i] > 0.5 and              # bullish weekly trend
                rsi[i] < 35 and                                # oversold
                volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: price near weekly VWAP (pullback) in bearish weekly trend + RSI overbought + volume
            elif (close[i] >= weekly_vwap_aligned[i] * 0.995 and  # within 0.5% below VWAP
                  weekly_trend_aligned[i] < 0.5 and              # bearish weekly trend
                  rsi[i] > 65 and                                # overbought
                  volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below weekly VWAP or RSI overbought
            if close[i] < weekly_vwap_aligned[i] * 0.995 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above weekly VWAP or RSI oversold
            if close[i] > weekly_vwap_aligned[i] * 1.005 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals