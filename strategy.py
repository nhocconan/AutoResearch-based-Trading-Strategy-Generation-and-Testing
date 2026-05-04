#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX Trend Filter + Volume Confirmation
# Williams %R identifies overbought/oversold conditions (thresholds: -10 for overbought, -90 for oversold)
# ADX > 25 confirms trend strength; we trade only in the direction of the 1d trend to avoid whipsaws
# Volume > 1.5x 20-period EMA ensures breakout/breakdown has conviction
# Discrete sizing (0.25) to balance return and risk. Target: 12-37 trades/year on 6h.
# Works in bull markets via buying oversold dips in uptrends and bear markets via selling overbought rallies in downtrends.

name = "6h_WilliamsR_Extreme_1dADX_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    period = 14
    highest_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ADX: Average Directional Index
    # +DM = max(0, high - previous high) if > max(0, previous low - low)
    # -DM = max(0, previous low - low) if > max(0, high - previous high)
    # TR = max(high - low, abs(high - previous close), abs(low - previous close))
    # +DI = 100 * smoothed +DM / ATR
    # -DI = 100 * smoothed -DM / ATR
    # ADX = 100 * smoothed |+DI - -DI| / (+DI + -DI)
    
    # Calculate directional movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Prepend 0 for first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing (Wilder's smoothing: alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # first value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    plus_di = 100 * wilders_smoothing(plus_dm, period_adx) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period_adx) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Avoid division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = wilders_smoothing(dx, period_adx)
    
    # Williams %R extremes: oversold < -90, overbought > -10
    williams_oversold = williams_r < -90
    williams_overbought = williams_r > -10
    
    # ADX trend filter: ADX > 25 indicates strong trend
    strong_trend = adx > 25
    # Determine trend direction using +DI and -DI
    uptrend = (plus_di > minus_di) & strong_trend
    downtrend = (minus_di > plus_di) & strong_trend
    
    # Align 1d indicators to 6h timeframe
    williams_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_oversold.astype(float))
    williams_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_overbought.astype(float))
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_oversold_aligned[i]) or np.isnan(williams_overbought_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -90) AND 1d uptrend AND volume spike
            if (williams_oversold_aligned[i] > 0.5 and 
                uptrend_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -10) AND 1d downtrend AND volume spike
            elif (williams_overbought_aligned[i] > 0.5 and 
                  downtrend_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 (exit oversold) OR 1d trend changes to downtrend
            if (williams_oversold_aligned[i] < 0.5 or 
                downtrend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 (exit overbought) OR 1d trend changes to uptrend
            if (williams_overbought_aligned[i] < 0.5 or 
                uptrend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals