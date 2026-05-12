#!/usr/bin/env python3
# 1d_1W_Camarilla_R4S4_Breakout_1wTrend_Volume
# Hypothesis: Breakouts at weekly Camarilla R4/S4 levels with 1w trend filter and volume confirmation.
# Weekly timeframe filters noise and captures major trend moves. Works in bull/bear:
# - Buy when price breaks above weekly R4 in bullish weekly trend with volume spike
# - Sell when price breaks below weekly S4 in bearish weekly trend with volume spike
# Targets 10-30 trades/year on 1d timeframe to avoid fee drag. Focus on BTC/ETH.

name = "1d_1W_Camarilla_R4S4_Breakout_1wTrend_Volume"
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

    # Get 1w data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate EMA(34) for 1w trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)

    # Get 1w data for Camarilla R4/S4 levels (from previous week)
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1w OHLC
    # Using previous week's data to avoid look-ahead
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values

    # Camarilla R4 and S4 levels (outer bands)
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2

    # Align Camarilla levels to 1d timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from EMA(34)
        bullish_trend = close[i] > ema_34_aligned[i]
        bearish_trend = close[i] < ema_34_aligned[i]

        if position == 0:
            # LONG: Break above weekly Camarilla R4 in bullish trend with volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and bullish_trend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly Camarilla S4 in bearish trend with volume confirmation
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