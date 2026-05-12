# 12h_1D_Camarilla_R3S3_Breakout_Trend_Volume_Filtered
# Hypothesis: Breakout at daily Camarilla R3/S3 levels with volume confirmation, 1d trend filter, and
# additional Choppy Market Index filter to reduce whipsaws in ranging markets. Designed to work in both
# bull and bear markets by requiring volume confirmation, trend alignment, and low-chop conditions.
# Targets 12-37 trades/year on 12h timeframe to avoid excessive fee drag.

name = "12h_1D_Camarilla_R3S3_Breakout_Trend_Volume_Filtered"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate Choppy Market Index (CMI) on 1d timeframe
    # CMI = 100 * (sum of true ranges over n periods) / (n * (highest high - lowest low over n periods))
    # Values near 0 indicate strong trend, values near 100 indicate choppy/ranging market
    lookback_period = 14
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr_sum = true_range.rolling(window=lookback_period, min_periods=lookback_period).sum()
    highest_high = df_1d['high'].rolling(window=lookback_period, min_periods=lookback_period).max()
    lowest_low = df_1d['low'].rolling(window=lookback_period, min_periods=lookback_period).min()
    cmi = 100 * (atr_sum / (lookback_period * (highest_high - lowest_low)))
    cmi_values = cmi.fillna(100).values  # Fill NaN with 100 (max chop) for safety
    cmi_aligned = align_htf_to_ltf(prices, df_1d, cmi_values)

    # Calculate Camarilla R3 and S3 levels from previous 1d OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4

    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Volume confirmation: current volume > 1.8x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ok[i]) or
            np.isnan(cmi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Chop filter: only trade when market is not too choppy (CMI < 50)
        low_chop = cmi_aligned[i] < 50

        if position == 0:
            # LONG: Break above Camarilla R3 with bullish trend, volume confirmation, and low chop
            if (close[i] > camarilla_r3_aligned[i] and bullish_trend and volume_ok[i] and low_chop):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S3 with bearish trend, volume confirmation, and low chop
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