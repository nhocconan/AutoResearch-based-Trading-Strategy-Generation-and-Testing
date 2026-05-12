#!/usr/bin/env python3
"""
1d_Ichimoku_Kumo_Twist_WeeklyTrend
Hypothesis: Trade Ichimoku Kumo twists (Senkou Span A/B cross) on daily timeframe when confirmed by weekly trend (price above/below weekly Kumo) and volume spike. Enters on Kumo twist in direction of weekly trend, exits on opposite twist. Weekly trend filter reduces whipsaws in ranging markets, volume spike confirms momentum. Works in bull/bear by following weekly trend. Target: 15-30 trades/year.
"""

name = "1d_Ichimoku_Kumo_Twist_WeeklyTrend"
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

    # Get weekly data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)

    # Calculate Ichimoku components on daily
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)

    # Kumo twist signals: Senkou A crosses Senkou B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_shift = np.roll(senkou_a, 1)
    senkou_b_shift = np.roll(senkou_b, 1)
    senkou_a_shift[0] = np.nan
    senkou_b_shift[0] = np.nan
    
    bullish_twist = (senkou_a > senkou_b) & (senkou_a_shift <= senkou_b_shift)
    bearish_twist = (senkou_a < senkou_b) & (senkou_a_shift >= senkou_b_shift)

    # Weekly trend: price above/below weekly Kumo
    # Calculate weekly Ichimoku for trend filter
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values

    # Weekly Tenkan-sen (9-period)
    w_period9_high = pd.Series(whigh).rolling(window=9, min_periods=9).max().values
    w_period9_low = pd.Series(wlow).rolling(window=9, min_periods=9).min().values
    w_tenkan = (w_period9_high + w_period9_low) / 2

    # Weekly Kijun-sen (26-period)
    w_period26_high = pd.Series(whigh).rolling(window=26, min_periods=26).max().values
    w_period26_low = pd.Series(wlow).rolling(window=26, min_periods=26).min().values
    w_kijun = (w_period26_high + w_period26_low) / 2

    # Weekly Senkou Span A
    w_senkou_a = ((w_tenkan + w_kijun) / 2)
    # Weekly Senkou Span B (52-period)
    w_period52_high = pd.Series(whigh).rolling(window=52, min_periods=52).max().values
    w_period52_low = pd.Series(wlow).rolling(window=52, min_periods=52).min().values
    w_senkou_b = ((w_period52_high + w_period52_low) / 2)

    # Align weekly Kumo to daily
    w_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, w_senkou_a)
    w_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, w_senkou_b)

    # Weekly trend: price above weekly Kumo (bullish) or below (bearish)
    weekly_trend_bullish = close > np.maximum(w_senkou_a_aligned, w_senkou_b_aligned)
    weekly_trend_bearish = close < np.minimum(w_senkou_a_aligned, w_senkou_b_aligned)

    # Volume spike: current > 2.0x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after weekly Kumo warmup
        if (np.isnan(bullish_twist[i]) or np.isnan(bearish_twist[i]) or
            np.isnan(weekly_trend_bullish[i]) or np.isnan(weekly_trend_bearish[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: bullish twist + weekly trend bullish + volume spike
            if bullish_twist[i] and weekly_trend_bullish[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish twist + weekly trend bearish + volume spike
            elif bearish_twist[i] and weekly_trend_bearish[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish twist (opposite signal)
            if bearish_twist[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish twist (opposite signal)
            if bullish_twist[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals