#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_Trend_VolumeS_v3
Hypothesis: Daily timeframe strategy trading breakouts from daily Camarilla R1/S1 levels in the direction of weekly trend, with volume confirmation and volatility filter to reduce whipsaws. Uses 1d for entries/exits and 1w for trend filter to capture multi-day moves while avoiding overtrading. Designed for 30-80 trades/year on BTC/ETH.
"""

name = "1d_Camarilla_R1_S1_Breakout_Trend_VolumeS_v3"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Get daily data for Camarilla levels and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels (R1, S1) from previous day
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12

    # Align Camarilla levels to daily timeframe (already aligned but keep for consistency)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate daily ATR(14) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values

    # Calculate daily volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(atr_ma[i]) or
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volatility filter: only trade when ATR is above its 50-day average
        vol_filter = atr[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else False
        
        # Volume confirmation: require 1.5x average volume
        vol_confirm = volume[i] > volume_sma20[i] * 1.5 if not np.isnan(volume_sma20[i]) else False

        if position == 0:
            # LONG: Breakout above R1 in weekly uptrend with volume and volatility confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and 
                vol_confirm and vol_filter):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 in weekly downtrend with volume and volatility confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  vol_confirm and vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly EMA20 (trend change) or volatility drops
            if close[i] < ema20_1w_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly EMA20 (trend change) or volatility drops
            if close[i] > ema20_1w_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals