#!/usr/bin/env python3
"""
4h_RangeBound_MeanReversion_RSI_Bollinger
Hypothesis: In ranging markets (Bollinger Band Width < 30th percentile), RSI extremes at Bollinger Bands
provide high-probability mean reversion entries. Uses 1d ADX < 20 to confirm range regime and avoid trending
markets. Works in both bull and bear markets as ranging behavior occurs in all regimes.
Target: 20-50 trades/year with low turnover to minimize fee drag.
"""

name = "4h_RangeBound_MeanReversion_RSI_Bollinger"
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

    # Get 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 14-period ADX for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().fillna(0).values
        return adx

    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # Calculate Bollinger Bands (20, 2) and %B
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    # Percentile rank of BB width
    bb_width_rank = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    # Percent B: (close - lower) / (upper - lower)
    percent_b = (close - lower_bb) / (upper_bb - lower_bb)
    percent_b = np.where((upper_bb - lower_bb) != 0, percent_b, 0.5)

    # Calculate RSI (14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(close, 14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        adx = adx_1d_aligned[i]
        bb_rank = bb_width_rank[i]
        pb = percent_b[i]
        rsi_val = rsi[i]

        # Skip if any required data is NaN
        if (np.isnan(adx) or np.isnan(bb_rank) or 
            np.isnan(pb) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Range regime: ADX < 20 (no trend) AND BB width in lower 30% (squeeze)
        if adx >= 20 or bb_rank > 0.3:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 30 and price at or below lower Bollinger Band (%B <= 0)
            if rsi_val < 30 and pb <= 0:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 and price at or above upper Bollinger Band (%B >= 1)
            elif rsi_val > 70 and pb >= 1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 or price at upper band (%B >= 0.8)
            if rsi_val > 50 or pb >= 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 or price at lower band (%B <= 0.2)
            if rsi_val < 50 or pb <= 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals