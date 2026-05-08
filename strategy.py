#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily ADX(14) > 25 trend filter with weekly EMA(34) alignment for stronger trend confirmation.
# We go long when daily ADX > 25 (trending market) AND price > weekly EMA(34) (bullish trend).
# We go short when daily ADX > 25 AND price < weekly EMA(34) (bearish trend).
# Uses 1d timeframe to target 7-25 trades/year, avoiding excessive frequency.
# ADX filters out ranging markets, improving win rate in both bull and bear regimes.
# Weekly EMA ensures we trade with higher timeframe momentum, reducing whipsaws.

name = "1d_ADX25_WeeklyEMA34_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate ADX(14) on daily data
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, high - prev_close, prev_close - low)
    # +DM14 = smoothed +DM over 14 periods
    # -DM14 = smoothed -DM over 14 periods
    # +DI14 = 100 * smoothed +DM14 / ATR14
    # -DI14 = 100 * smoothed -DM14 / ATR14
    # DX = 100 * |+DI14 - -DI14| / (+DI14 + -DI14)
    # ADX = smoothed DX over 14 periods
    
    # Calculate directional movements
    high_diff = high[1:] - high[:-1]
    low_diff = low[:-1] - low[1:]
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - low[:-1])
    tr3 = np.abs(low[1:] - high[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    # Pad arrays to match original length (first element has no previous)
    plus_dm_padded = np.concatenate([[0.0], plus_dm])
    minus_dm_padded = np.concatenate([[0.0], minus_dm])
    tr_padded = np.concatenate([[0.0], tr])
    
    # Smooth with Wilder's method
    atr14 = wilders_smoothing(tr_padded, 14)
    plus_di14 = 100 * wilders_smoothing(plus_dm_padded, 14) / atr14
    minus_di14 = 100 * wilders_smoothing(minus_dm_padded, 14) / atr14
    
    # Avoid division by zero
    di_sum = plus_di14 + minus_di14
    dx = np.where(di_sum > 0, 100 * np.abs(plus_di14 - minus_di14) / di_sum, 0.0)
    adx = wilders_smoothing(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        if position == 0:
            # Enter long: ADX > 25 (trending) AND price > weekly EMA (bullish)
            if adx_val > 25.0 and close[i] > ema34_1w_val:
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 (trending) AND price < weekly EMA (bearish)
            elif adx_val > 25.0 and close[i] < ema34_1w_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX drops below 20 (losing trend) OR price crosses below weekly EMA
            if adx_val < 20.0 or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX drops below 20 (losing trend) OR price crosses above weekly EMA
            if adx_val < 20.0 or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals