#!/usr/bin/env python3
# 1d_RSI_Divergence_Volume_Trend
# Hypothesis: Use 1d RSI divergence (bullish/bearish) with volume confirmation and weekly trend filter for high-probability reversals.
# Works in bull markets (buy bullish divergence) and bear markets (sell bearish divergence) by filtering counter-trend trades.
# Low-frequency signals reduce fee drag; RSI divergence captures exhaustion moves.

name = "1d_RSI_Divergence_Volume_Trend"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate daily RSI(14)
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Find RSI swing points for divergence detection
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    def find_swing_points(arr, window=5):
        # Simple peak/trough detection
        peaks = np.zeros_like(arr, dtype=bool)
        troughs = np.zeros_like(arr, dtype=bool)
        for i in range(window, len(arr) - window):
            if arr[i] == np.max(arr[i-window:i+window+1]):
                peaks[i] = True
            if arr[i] == np.min(arr[i-window:i+window+1]):
                troughs[i] = True
        return peaks, troughs

    rsi_peaks, rsi_troughs = find_swing_points(rsi, window=5)
    price_peaks, price_troughs = find_swing_points(close, window=5)

    # Bullish divergence: price trough + RSI higher trough
    bullish_div = np.zeros(n, dtype=bool)
    # Bearish divergence: price peak + RSI lower peak
    bearish_div = np.zeros(n, dtype=bool)

    last_price_trough = -1
    last_rsi_trough = -1
    last_price_peak = -1
    last_rsi_peak = -1

    for i in range(n):
        if price_troughs[i]:
            if last_price_trough != -1 and rsi[i] > rsi[last_rsi_trough] and close[i] < close[last_price_trough]:
                bullish_div[i] = True
            last_price_trough = i
            last_rsi_trough = i
        if price_peaks[i]:
            if last_price_peak != -1 and rsi[i] < rsi[last_rsi_peak] and close[i] > close[last_price_peak]:
                bearish_div[i] = True
            last_price_peak = i
            last_rsi_peak = i

    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_conf[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish divergence with volume confirmation and uptrend (price > weekly EMA50)
            if bullish_div[i] and volume_conf[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence with volume confirmation and downtrend (price < weekly EMA50)
            elif bearish_div[i] and volume_conf[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly EMA50 or RSI > 70 (overbought)
            if close[i] < ema50_1w_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly EMA50 or RSI < 30 (oversold)
            if close[i] > ema50_1w_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals