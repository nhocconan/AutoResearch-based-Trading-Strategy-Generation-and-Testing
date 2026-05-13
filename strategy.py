#!/usr/bin/env python3
# 4h_PriceAction_InsideBar_1dTrend_Volume
# Hypothesis: Inside bars on 4h indicate consolidation and low volatility. A breakout of the inside bar's high/low with 1d trend alignment and volume confirmation captures institutional participation. Works in bull markets (breakouts above inside bar high in uptrend) and bear markets (breakdowns below inside bar low in downtrend). Inside bars filter out false breakouts, reducing whipsaw. Target: 20-40 trades/year per symbol to minimize fee drag.

name = "4h_PriceAction_InsideBar_1dTrend_Volume"
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

    # Get 1d data for trend
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Inside bar detection: current bar's high <= previous bar's high AND low >= previous bar's low
    inside_bar = (high <= np.roll(high, 1)) & (low >= np.roll(low, 1))
    inside_bar[0] = False  # First bar has no previous
    
    # Inside bar high and low (reference for breakout)
    inside_bar_high = np.where(inside_bar, high, np.nan)
    inside_bar_low = np.where(inside_bar, low, np.nan)
    
    # Forward fill inside bar levels to use until next inside bar
    inside_bar_high_ffill = pd.Series(inside_bar_high).ffill().values
    inside_bar_low_ffill = pd.Series(inside_bar_low).ffill().values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(inside_bar_high_ffill[i]) or 
            np.isnan(inside_bar_low_ffill[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > inside bar high + 1d uptrend + volume spike
            if close[i] > inside_bar_high_ffill[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < inside bar low + 1d downtrend + volume spike
            elif close[i] < inside_bar_low_ffill[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below inside bar low or trend reversal
            if close[i] < inside_bar_low_ffill[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above inside bar high or trend reversal
            if close[i] > inside_bar_high_ffill[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals