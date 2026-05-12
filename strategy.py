#!/usr/bin/env python3
# 4h_KAMA_Trend_Follow_With_Chop_Filter
# Hypothesis: KAMA adapts to market noise - in trending markets it tracks price closely, in ranging markets it stays flat.
# Combined with Choppiness Index regime filter: only trade when market is trending (CHOP < 38.2).
# Uses 12h timeframe for trend direction to reduce whipsaw. Low trade frequency expected.
# Works in bull markets by following uptrend, in bear markets by following downtrend.
# Avoids whipsaws in ranging markets via CHOP filter.

name = "4h_KAMA_Trend_Follow_With_Chop_Filter"
timeframe = "4h"
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

    # Get 12h data for KAMA trend filter (primary trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    close_12h = df_12h['close'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |Change| / Volatility, where Change = |close - close[10]|, Volatility = sum|diff| over 10 periods
    change = np.abs(close_12h - np.roll(close_12h, 10))
    volatility = np.sum(np.abs(np.diff(close_12h, axis=0)), axis=0)  # temporary, will compute properly below
    # Recalculate volatility properly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_12h)
    for i in range(10, len(close_12h)):
        volatility[i] = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants: fastest EMA = 2/(2+1) = 0.67, slowest = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)

    # Get 4h data for Choppiness Index calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate Choppiness Index (CHOP)
    # True Range = max(high-low, |high-previous close|, |low-previous close|)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_4h[0] - low_4h[0]  # First TR
    
    # Sum of TRUE RANGE over 14 periods
    tr_sum = np.zeros_like(close_4h)
    for i in range(14, len(close_4h)):
        tr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_hh = np.zeros_like(close_4h)
    min_ll = np.zeros_like(close_4h)
    for i in range(14, len(close_4h)):
        max_hh[i] = np.max(high_4h[i-13:i+1])
        min_ll[i] = np.min(low_4h[i-13:i+1])
    
    # CHOP = 100 * log10(sum(tr14) / (max(hh14) - min(ll14))) / log10(14)
    range_hl = max_hh - min_ll
    # Avoid division by zero and log of zero
    chop = np.zeros_like(close_4h)
    mask = (range_hl > 0) & (tr_sum > 0)
    chop[mask] = 100 * np.log10(tr_sum[mask] / range_hl[mask]) / np.log10(14)
    
    # Align CHOP to 4h timeframe (already on 4h, so just use directly)
    chop_aligned = chop  # Already calculated on 4h data

    # Calculate volume confirmation (1.5x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA (uptrend) AND market is trending (CHOP < 38.2) AND volume confirmation
            if (close[i] > kama_aligned[i] and 
                chop_aligned[i] < 38.2 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) AND market is trending (CHOP < 38.2) AND volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  chop_aligned[i] < 38.2 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR market becomes ranging (CHOP > 61.8)
            if (close[i] < kama_aligned[i]) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR market becomes ranging (CHOP > 61.8)
            if (close[i] > kama_aligned[i]) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals