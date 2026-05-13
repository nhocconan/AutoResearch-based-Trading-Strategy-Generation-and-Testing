#!/usr/bin/env python3
# 1d_WeeklyCandlePattern_TrendFilter
# Hypothesis: Weekly candlestick patterns (engulfing) on the weekly chart combined with daily trend filter and volume confirmation.
# Uses bullish/bearish engulfing patterns on weekly timeframe, filtered by daily EMA50 trend and volume > 1.5x 20-day average.
# Designed for low-frequency, high-conviction trades that work in both bull and bear markets by capturing weekly momentum shifts.

name = "1d_WeeklyCandlePattern_TrendFilter"
timeframe = "1d"
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

    # Get weekly data for engulfing patterns
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly bullish and bearish engulfing patterns
    weekly_open = df_weekly['open'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Bullish engulfing: current weekly candle closes above previous weekly open AND opens below previous weekly close
    bullish_engulf = (weekly_close > weekly_open[:-1]) & (weekly_open < weekly_close[:-1])
    # Bearish engulfing: current weekly candle closes below previous weekly open AND opens above previous weekly close
    bearish_engulf = (weekly_close < weekly_open[:-1]) & (weekly_open > weekly_close[:-1])
    
    # Prepend False for first week (no previous week)
    bullish_engulf = np.concatenate([[False], bullish_engulf])
    bearish_engulf = np.concatenate([[False], bearish_engulf])
    
    # Align weekly patterns to daily timeframe (with proper delay for weekly close)
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_weekly, bullish_engulf.astype(float))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_weekly, bearish_engulf.astype(float))

    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    ema_50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)

    # Volume filter: >1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_daily_aligned[i]) or 
            np.isnan(bullish_engulf_aligned[i]) or 
            np.isnan(bearish_engulf_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish weekly engulfing + price above daily EMA50 (uptrend) + volume spike
            if (bullish_engulf_aligned[i] == 1.0 and 
                close[i] > ema_50_daily_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish weekly engulfing + price below daily EMA50 (downtrend) + volume spike
            elif (bearish_engulf_aligned[i] == 1.0 and 
                  close[i] < ema_50_daily_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish weekly engulfing or price below daily EMA50 (trend change)
            if (bearish_engulf_aligned[i] == 1.0 or close[i] < ema_50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish weekly engulfing or price above daily EMA50 (trend change)
            if (bullish_engulf_aligned[i] == 1.0 or close[i] > ema_50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals