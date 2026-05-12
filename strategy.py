#!/usr/bin/env python3
# 12h_1D_Camarilla_R4S4_Breakout_Trend_Volume
# Hypothesis: Breakouts at daily Camarilla R4/S4 levels with 1d EMA trend filter and volume confirmation.
# Works in bull/bear: Buy when price breaks above R4 in bullish trend (close > EMA34) with volume spike;
# Sell when price breaks below S4 in bearish trend (close < EMA34) with volume spike.
# Targets 15-35 trades/year on 12h timeframe to avoid fee drag. Focus on BTC/ETH.

name = "12h_1D_Camarilla_R4S4_Breakout_Trend_Volume"
timeframe = "12h"
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

    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)

    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)

    # Get 1d data for Camarilla R4/S4 levels (from previous day)
    if len(df_1d) < 40:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1d OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Camarilla R4 and S4 levels (outer bands)
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2

    # Align Camarilla levels to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from EMA34
        bullish_trend = close[i] > ema34_aligned[i]
        bearish_trend = close[i] < ema34_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R4 in bullish trend with volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and bullish_trend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S4 in bearish trend with volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and bearish_trend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R4 or trend turns bearish
            if close[i] < camarilla_r4_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S4 or trend turns bullish
            if close[i] > camarilla_s4_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals