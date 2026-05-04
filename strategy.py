#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and ATR-based position sizing
# Long when price breaks above 20-day Donchian high AND 1w bullish trend (close > EMA50) AND ATR(14) < 0.03 * close (low volatility regime)
# Short when price breaks below 20-day Donchian low AND 1w bearish trend (close < EMA50) AND ATR(14) < 0.03 * close
# Uses 1w EMA50 for trend filter to reduce whipsaw, targeting 15-25 trades/year on 1d.
# ATR volatility filter ensures trades occur only in stable market conditions, reducing false breakouts.
# Works in bull markets via longs in bullish 1w trend regime and bear markets via shorts in bearish 1w trend regime.

name = "1d_Donchian20_1wTrend_ATRVolFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1w = close_1w > ema_50_1w
    trend_bearish_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate 20-day Donchian channels from 1d data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: ATR < 3% of price (low volatility regime)
    vol_filter = atr_14 < (0.03 * close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 1w bullish trend AND low volatility
            if (close[i] > highest_high[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND 1w bearish trend AND low volatility
            elif (close[i] < lowest_low[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR 1w trend turns bearish
            if (close[i] < lowest_low[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR 1w trend turns bullish
            if (close[i] > highest_high[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals