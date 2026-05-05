#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ATR volatility filter and 1w EMA trend filter
# Long when price breaks above 12h Camarilla R3 level AND 1w EMA50 > EMA200 (bullish trend) AND ATR(14) < 0.5 * ATR(50) (low volatility)
# Short when price breaks below 12h Camarilla S3 level AND 1w EMA50 < EMA200 (bearish trend) AND ATR(14) < 0.5 * ATR(50) (low volatility)
# Exit when price crosses 12h Camarilla pivot point (mean reversion) OR volatility expands (ATR(14) > ATR(50))
# Uses 12h primary timeframe with 1w HTF for EMA trend filter and 1d for ATR volatility filter and Camarilla levels
# Volatility contraction breakouts capture explosive moves after consolidation, working in both bull and bear markets
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1wEMA_Trend_ATR_Vol"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Wilder's smoothing for ATR
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            else:
                result[i] = np.nan
        return result
    
    atr_14 = wilder_smooth(tr, 14)
    atr_50 = wilder_smooth(tr, 50)
    
    # Align ATR to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Calculate 1w EMA50 and EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Calculate 1d Camarilla levels (based on previous 1d bar)
    camarilla_r3 = close_1d + (1.1 * (high_1d - low_1d) / 2)
    camarilla_s3 = close_1d - (1.1 * (high_1d - low_1d) / 2)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3  # Standard pivot point
    
    # Align to 12h timeframe (using previous 1d bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: ATR(14) < 0.5 * ATR(50) (low volatility)
        vol_filter = atr_14_aligned[i] < (0.5 * atr_50_aligned[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND bullish trend AND low volatility
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_50_aligned[i] > ema_200_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND bearish trend AND low volatility
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion) OR volatility expands (ATR(14) > ATR(50))
            if close[i] < camarilla_pivot_aligned[i] or atr_14_aligned[i] > atr_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion) OR volatility expands (ATR(14) > ATR(50))
            if close[i] > camarilla_pivot_aligned[i] or atr_14_aligned[i] > atr_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals