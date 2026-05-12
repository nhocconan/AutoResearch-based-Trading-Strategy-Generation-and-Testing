#!/usr/bin/env python3
"""
6h_RSI200_Trend_Breakout_With_Volume
Hypothesis: On 6h timeframe, RSI(14) cross above 60 with price above 200-period EMA
and volume > 1.5x 20-period average signals strong momentum continuation in bull markets;
RSI < 40 with price below 200 EMA and volume surge signals bearish continuation.
Uses 1d ADX > 25 to filter trending regimes only, avoiding choppy markets.
Targets 12-37 trades/year (50-150 total over 4 years) with low turnover to minimize fee drag.
Works in bull via momentum breaks above resistance and bear via breakdowns below support.
"""

name = "6h_RSI200_Trend_Breakout_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d ADX(14) for trend filter
    # ADX calculation: +DM, -DM, TR, then DX, then smoothed ADX
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value

        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm[0] = 0.0
        minus_dm[0] = 0.0

        # Smoothed values
        def smooth(values, period):
            smoothed = np.zeros_like(values)
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
            return smoothed

        tr_smooth = smooth(tr, period)
        plus_dm_smooth = smooth(plus_dm, period)
        minus_dm_smooth = smooth(minus_dm, period)

        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth

        # DX and ADX
        dx = np.zeros_like(close)
        dx[tr_smooth != 0] = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = smooth(dx, period)
        return adx

    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # Calculate 200-period EMA on 6h close
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values

    # Calculate RSI(14) on 6h close
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period+1])
        avg_loss[period] = np.mean(loss[:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(close, 14)

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Get aligned values for current 6h bar
        adx = adx_1d_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(adx) or np.isnan(ema200[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: only trade when ADX > 25 (trending market)
        if adx <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 60 + price above EMA200 + volume surge
            if (rsi[i] > 60 and 
                close[i] > ema200[i] and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 40 + price below EMA200 + volume surge
            elif (rsi[i] < 40 and 
                  close[i] < ema200[i] and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 50 or price below EMA200
            if (rsi[i] < 50 or close[i] < ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 50 or price above EMA200
            if (rsi[i] > 50 or close[i] > ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals