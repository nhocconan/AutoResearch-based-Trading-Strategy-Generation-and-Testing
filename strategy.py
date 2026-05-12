#!/usr/bin/env python3

# 6h_1W_1D_Camarilla_R3S3_Breakout_Trend
# Hypothesis: Breakout above Camarilla R3 or below S3 on daily timeframe, with weekly trend filter and volume confirmation.
# Uses weekly ADX > 20 to confirm trending market and daily volume > 1.5x 20-bar average for confirmation.
# In trending markets, Camarilla R3/S3 act as breakout levels; in ranging markets, they act as reversal levels.
# Weekly trend filter ensures we only take breakouts in the direction of the higher timeframe trend.
# Targets 15-30 trades/year to minimize fee drag while capturing significant moves.

name = "6h_1W_1D_Camarilla_R3S3_Breakout_Trend"
timeframe = "6h"
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

    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Camarilla calculation: based on previous day's range
    R3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    S3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    R4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    S4 = close_1d - 1.1 * (high_1d - low_1d) / 2

    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)

    # Weekly trend filter: ADX > 20 indicates trending market
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Calculate +DI and -DI
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr

    # Calculate ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)

    # Weekly trend direction: +DI > -DI for uptrend, -DI > +DI for downtrend
    plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di)
    bullish_trend = plus_di_aligned > minus_di_aligned
    bearish_trend = minus_di_aligned > plus_di_aligned

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(bullish_trend[i]) or
            np.isnan(bearish_trend[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only take trades when weekly ADX > 20 (trending market)
        if adx_aligned[i] > 20:
            if position == 0:
                # LONG: Price breaks above R3 with bullish weekly trend and volume confirmation
                if close[i] > R3_aligned[i] and bullish_trend[i] and volume_ok[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price breaks below S3 with bearish weekly trend and volume confirmation
                elif close[i] < S3_aligned[i] and bearish_trend[i] and volume_ok[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Price closes below S3 or trend turns bearish
                if close[i] < S3_aligned[i] or not bullish_trend[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Price closes above R3 or trend turns bullish
                if close[i] > R3_aligned[i] or not bearish_trend[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging market (ADX <= 20), fade at R3/S3
            if position == 0:
                # LONG: Price rejects S3 with volume confirmation
                if close[i] < S3_aligned[i] and close[i] > S4_aligned[i] and volume_ok[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price rejects R3 with volume confirmation
                elif close[i] > R3_aligned[i] and close[i] < R4_aligned[i] and volume_ok[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Price reaches R3 or reverses
                if close[i] > R3_aligned[i] or close[i] < S3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Price reaches S3 or reverses
                if close[i] < S3_aligned[i] or close[i] > R3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25

    return signals