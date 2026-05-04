#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# Elder Ray: Bull Power = High - EMA13(Close), Bear Power = EMA13(Close) - Low
# Long when Bull Power > 0 AND 12h ADX > 25 (trending market)
# Short when Bear Power > 0 AND 12h ADX > 25 (trending market)
# Uses 6h for entry precision, 12h for regime filter to avoid ranging markets.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-30 trades/year.
# Works in bull markets via longs in strong uptrends and bear markets via shorts in strong downtrends.

name = "6h_ElderRay_12hADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for ADX regime filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for alignment
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        def ema_smoothing(values, period):
            ema = np.full_like(values, np.nan, dtype=float)
            if len(values) >= period:
                # First value is simple average
                ema[period-1] = np.nanmean(values[:period])
                # Subsequent values
                alpha = 2 / (period + 1)
                for i in range(period, len(values)):
                    if not np.isnan(values[i]):
                        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
                    else:
                        ema[i] = ema[i-1]
            return ema
        
        atr = ema_smoothing(tr, period)
        dm_plus_ema = ema_smoothing(dm_plus, period)
        dm_minus_ema = ema_smoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_ema / atr
        di_minus = 100 * dm_minus_ema / atr
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = ema_smoothing(dx, period)
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    trending_12h = adx_12h > 25  # Strong trend when ADX > 25
    
    # Align 12h trend to 6h timeframe
    trending_12h_aligned = align_htf_to_ltf(prices, df_12h, trending_12h.astype(float))
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h  # Bull Power = High - EMA13
    bear_power = ema_13_6h - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(trending_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND 12h trending (ADX > 25)
            if (bull_power[i] > 0 and trending_12h_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 AND 12h trending (ADX > 25)
            elif (bear_power[i] > 0 and trending_12h_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR 12h trend weakens (ADX <= 25)
            if (bull_power[i] <= 0 or trending_12h_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 OR 12h trend weakens (ADX <= 25)
            if (bear_power[i] <= 0 or trending_12h_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals