#!/usr/bin/env python3
# 12h_1W_1D_Camarilla_R3S3_Breakout_Trend_Volume_Regime
# Hypothesis: Breakout at weekly and daily combined strong support/resistance levels (R3/S3) with volume confirmation, trend filter from weekly EMA, and a chop regime filter to avoid ranging markets. Designed to work in both bull and bear markets by requiring trend alignment and low-chop conditions. Targets 15-25 trades/year on 12h timeframe to minimize fee drag.

name = "12h_1W_1D_Camarilla_R3S3_Breakout_Trend_Volume_Regime"
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

    # Get weekly and daily data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)

    # Weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate Choppy Market Index (CMI) on weekly timeframe
    lookback_period = 14
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr_sum = true_range.rolling(window=lookback_period, min_periods=lookback_period).sum()
    highest_high = df_1w['high'].rolling(window=lookback_period, min_periods=lookback_period).max()
    lowest_low = df_1w['low'].rolling(window=lookback_period, min_periods=lookback_period).min()
    cmi = 100 * (atr_sum / (lookback_period * (highest_high - lowest_low)))
    cmi_values = cmi.fillna(100).values
    cmi_aligned = align_htf_to_ltf(prices, df_1w, cmi_values)

    # Combined strong levels: Weekly and Daily R3/S3 (average)
    # Weekly levels
    wc_prev_close = df_1w['close'].shift(1).values
    wc_prev_high = df_1w['high'].shift(1).values
    wc_prev_low = df_1w['low'].shift(1).values
    weekly_r3 = wc_prev_close + (wc_prev_high - wc_prev_low) * 1.1 / 4
    weekly_s3 = wc_prev_close - (wc_prev_high - wc_prev_low) * 1.1 / 4
    # Daily levels
    dc_prev_close = df_1d['close'].shift(1).values
    dc_prev_high = df_1d['high'].shift(1).values
    dc_prev_low = df_1d['low'].shift(1).values
    daily_r3 = dc_prev_close + (dc_prev_high - dc_prev_low) * 1.1 / 4
    daily_s3 = dc_prev_close - (dc_prev_high - dc_prev_low) * 1.1 / 4
    # Average for stronger confluence
    camarilla_r3 = (weekly_r3 + daily_r3) / 2
    camarilla_s3 = (weekly_s3 + daily_s3) / 2

    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)  # align to daily then to 12h
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Volume confirmation: current volume > 2.0x average of last 30 periods
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ok[i]) or
            np.isnan(cmi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below weekly EMA
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]
        
        # Chop filter: only trade when market is not too choppy (CMI < 40)
        low_chop = cmi_aligned[i] < 40

        if position == 0:
            # LONG: Break above combined R3 with bullish trend, volume confirmation, and low chop
            if (close[i] > camarilla_r3_aligned[i] and bullish_trend and volume_ok[i] and low_chop):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below combined S3 with bearish trend, volume confirmation, and low chop
            elif (close[i] < camarilla_s3_aligned[i] and bearish_trend and volume_ok[i] and low_chop):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend turns bearish or chop increases
            if close[i] < camarilla_r3_aligned[i] or not bullish_trend or not low_chop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend turns bullish or chop increases
            if close[i] > camarilla_s3_aligned[i] or not bearish_trend or not low_chop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals