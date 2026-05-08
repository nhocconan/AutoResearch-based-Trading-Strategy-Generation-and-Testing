#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray + momentum filter
# Uses Alligator (jaw/teeth/lips) for trend direction, Elder Ray (bull/bear power) for momentum,
# and RSI(14) for overbought/oversold filtering.
# Long when price > teeth, bull power > 0, and RSI < 70.
# Short when price < teeth, bear power < 0, and RSI > 30.
# Exit on opposite signal or when price crosses lips.
# Works in trending markets (Alligator aligned) and avoids chop via RSI extremes.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Alligator_ElderRay_RSI"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator (13,8,5) - smoothed with SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    def ema(arr, period):
        return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])
    
    # Get 1d data for higher timeframe trend filter (optional)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        ema_20_1d = ema(df_1d['close'].values, 20)
        ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
        trend_filter = ema_20_1d_aligned  # Use as dynamic threshold
    else:
        trend_filter = None
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price > teeth, bull power > 0, RSI < 70 (not overbought)
            if close[i] > teeth[i] and bull_power[i] > 0 and rsi[i] < 70:
                # Optional 1d trend filter: only long if price > 1d EMA20
                if trend_filter is None or close[i] > trend_filter[i]:
                    signals[i] = 0.25
                    position = 1
            # Short entry: price < teeth, bear power < 0, RSI > 30 (not oversold)
            elif close[i] < teeth[i] and bear_power[i] < 0 and rsi[i] > 30:
                if trend_filter is None or close[i] < trend_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price < lips OR opposite signal (price < teeth & bear power < 0)
            if close[i] < lips[i] or (close[i] < teeth[i] and bear_power[i] < 0 and rsi[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > lips OR opposite signal (price > teeth & bull power > 0)
            if close[i] > lips[i] or (close[i] > teeth[i] and bull_power[i] > 0 and rsi[i] < 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals