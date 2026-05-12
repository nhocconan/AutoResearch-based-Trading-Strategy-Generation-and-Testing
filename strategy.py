#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_200MA_Filter
Hypothesis: On daily timeframe, KAMA (Kaufman Adaptive Moving Average) adapts to market volatility,
providing robust trend identification. Combines with RSI(14) for momentum confirmation and
200-period SMA for long-term trend filter. In bull markets, KAMA above SMA200 with RSI>50
captures uptrends; in bear markets, KAMA below SMA200 with RSI<50 captures downtrends.
Uses weekly ADX to filter ranging markets (ADX<20) and avoid whipsaws. Targets 7-25 trades/year
(30-100 total over 4 years) with low turnover to minimize fee drag.
"""

name = "1d_KAMA_Trend_RSI_200MA_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    # Calculate weekly ADX(14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    up_move = np.concatenate([[np.nan], np.where((up_move > down_move) & (up_move > 0), up_move, 0)])
    down_move = np.concatenate([[np.nan], np.where((down_move > up_move) & (down_move > 0), down_move, 0)])

    # Directional Indicators
    plus_di = 100 * pd.Series(up_move).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di = 100 * pd.Series(down_move).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)

    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.concatenate([[np.nan], np.diff(close)]))
    volatility = np.sum(np.abs(np.concatenate([[np.nan], np.diff(close)])), axis=0) if False else None  # placeholder
    # Proper volatility calculation: sum of absolute changes over 10 periods
    volatility_sum = pd.Series(np.abs(np.concatenate([[np.nan], np.diff(close)])).rolling(window=10, min_periods=10).sum().values)
    change_sum = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_sum.values != 0, change_sum / volatility_sum.values, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)  # dummy df for alignment

    # Calculate RSI(14) on daily close
    delta = np.concatenate([[np.nan], np.diff(close)])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)

    # Calculate 200-period SMA on daily close
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    sma200_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), sma200)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Get aligned values for current daily bar
        adx = adx_1w_aligned[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        sma200_val = sma200_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(adx) or np.isnan(kama_val) or 
            np.isnan(rsi_val) or np.isnan(sma200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: only trade when ADX >= 20 (trending market)
        if adx < 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA above SMA200 + RSI > 50
            if (kama_val > sma200_val and rsi_val > 50):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA below SMA200 + RSI < 50
            elif (kama_val < sma200_val and rsi_val < 50):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA crosses below SMA200 or RSI < 40
            if (kama_val < sma200_val or rsi_val < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA crosses above SMA200 or RSI > 60
            if (kama_val > sma200_val or rsi_val > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals