#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_MeanReversion_Camarilla_Exit
Hypothesis: In trending markets (1d EMA50), use KAMA direction for trend following; in ranging markets (1d ADX < 20), use RSI mean reversion at extreme levels. Exit when price touches opposite Camarilla level (S1/R1) to capture mean reversion within the trend or range. Combines trend and mean reversion with regime filter to work in both bull and bear markets.
"""

name = "4h_KAMA_Trend_RSI_MeanReversion_Camarilla_Exit"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0])).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, length=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high, low, close, length=14):
    """Average Directional Index"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    tr = np.maximum(
        high - np.roll(high, 1),
        np.roll(high, 1) - np.roll(low, 1),
        np.roll(low, 1) - np.roll(close, 1)
    )
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean().values / \
              pd.Series(tr).ewm(alpha=1/length, adjust=False).mean().values
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean().values / \
               pd.Series(tr).ewm(alpha=1/length, adjust=False).mean().values
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = pd.Series(dx).ewm(alpha=1/length, adjust=False).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d indicators for regime filter
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # 1d ADX(14) for regime filter
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)

    # Calculate Camarilla levels from 1d data (R1, S1)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan

    camarilla_r1 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_s1 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12

    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Calculate KAMA(10,2,30) on 4h data
    kama = calculate_kama(close, 10, 2, 30)

    # Calculate RSI(14) on 4h data
    rsi = calculate_rsi(close, 14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(rsi[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: trending if ADX > 25, ranging if ADX < 20
        is_trending = adx_14_1d_aligned[i] > 25
        is_ranging = adx_14_1d_aligned[i] < 20

        if position == 0:
            if is_trending:
                # TREND FOLLOWING: KAMA direction
                if close[i] > kama[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # MEAN REVERSION: RSI extremes
                if rsi[i] < 30:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:  # Overbought
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition zone: no trade
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches Camarilla S1 (mean reversion target)
            if close[i] <= camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches Camarilla R1 (mean reversion target)
            if close[i] >= camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals