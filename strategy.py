#!/usr/bin/env python3
"""
1h_OrderBlock_Retest_4hTrend_1dVolFilter
Hypothesis: Price retests 4h order blocks (bullish/bearish) in direction of 1d EMA50 trend with volume confirmation. Designed for 15-30 trades/year on 1h timeframe to work in both bull and bear markets by using institutional order flow concepts and trend filtering.
"""

name = "1h_OrderBlock_Retest_4hTrend_1dVolFilter"
timeframe = "1h"
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

    # Get 4h data for order blocks (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Identify bullish and bearish order blocks on 4h
    # Bullish OB: bearish candle followed by strong bullish candle
    # Bearish OB: bullish candle followed by strong bearish candle
    body_4h = np.abs(close_4h - open_4h) if 'open_4h' in df_4h.columns else np.abs(close_4h - np.roll(close_4h, 1))
    # Simplified: look for strong directional candles
    bullish_4h = close_4h > np.roll(close_4h, 1)  # closing higher than previous close
    bearish_4h = close_4h < np.roll(close_4h, 1)  # closing lower than previous close
    
    # Bullish OB: previous candle bearish, current bullish with strong body
    bullish_ob = np.zeros(len(close_4h), dtype=bool)
    bearish_ob = np.zeros(len(close_4h), dtype=bool)
    
    for i in range(1, len(close_4h)):
        # Bullish OB: prev bearish, current bullish with strong close
        if bearish_4h[i-1] and bullish_4h[i]:
            body_size = abs(close_4h[i] - open_4h[i]) if 'open_4h' in df_4h.columns else abs(close_4h[i] - close_4h[i-1])
            avg_body = np.mean(np.abs(close_4h[max(0,i-5):i] - np.roll(close_4h[max(0,i-5):i], 1))) if i >= 5 else np.abs(close_4h[i] - close_4h[i-1])
            if body_size > avg_body * 1.5:  # strong bullish candle
                bullish_ob[i] = True
        # Bearish OB: prev bullish, current bearish with strong body
        elif bullish_4h[i-1] and bearish_4h[i]:
            body_size = abs(close_4h[i] - open_4h[i]) if 'open_4h' in df_4h.columns else abs(close_4h[i] - close_4h[i-1])
            avg_body = np.mean(np.abs(close_4h[max(0,i-5):i] - np.roll(close_4h[max(0,i-5):i], 1))) if i >= 5 else np.abs(close_4h[i] - close_4h[i-1])
            if body_size > avg_body * 1.5:  # strong bearish candle
                bearish_ob[i] = True

    # Store OB levels (high of bearish OB, low of bullish OB)
    ob_high = np.where(bearish_ob, high_4h, np.nan)
    ob_low = np.where(bullish_ob, low_4h, np.nan)
    
    # Forward fill OB levels until broken
    ob_high_series = pd.Series(ob_high)
    ob_low_series = pd.Series(ob_low)
    ob_high_ffill = ob_high_series.ffill().values
    ob_low_ffill = ob_low_series.ffill().values
    
    # Align to 1h timeframe
    ob_high_aligned = align_htf_to_ltf(prices, df_4h, ob_high_ffill)
    ob_low_aligned = align_htf_to_ltf(prices, df_4h, ob_low_ffill)

    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Get 1d volume for filter (20-period average)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        ob_high_val = ob_high_aligned[i]
        ob_low_val = ob_low_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_avg_val = vol_avg_20_1d_aligned[i]

        if np.isnan(ob_high_val) or np.isnan(ob_low_val) or np.isnan(ema50_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price retests bullish OB (support) + uptrend + volume confirmation
            if low[i] <= ob_low_val * 1.001 and close[i] > ob_low_val and close[i] > ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.20
                position = 1
            # SHORT: Price retests bearish OB (resistance) + downtrend + volume confirmation
            elif high[i] >= ob_high_val * 0.999 and close[i] < ob_high_val and close[i] < ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below OB low or reverses against trend
            if close[i] < ob_low_val * 0.995 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above OB high or reverses against trend
            if close[i] > ob_high_val * 1.005 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals