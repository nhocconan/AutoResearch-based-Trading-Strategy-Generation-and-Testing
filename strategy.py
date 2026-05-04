#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and ATR-based stoploss
# Long when price breaks above Donchian upper band AND 1d bullish trend (close > EMA50) AND ATR(14) < 0.03 * close (low volatility regime)
# Short when price breaks below Donchian lower band AND 1d bearish trend (close < EMA50) AND ATR(14) < 0.03 * close
# Uses 1d EMA50 for trend filter to reduce whipsaw, targeting 20-50 trades/year on 4h.
# ATR volatility filter avoids choppy markets, Donchian provides clear structure.
# Works in bull markets via longs in bullish 1d trend regime and bear markets via shorts in bearish 1d trend regime.
# Stoploss implemented as signal=0 when price moves against position by 2.5 * ATR(14)

name = "4h_Donchian20_1dTrend_ATRVolFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for HTF trend filter and ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1d = close_1d > ema_50_1d
    trend_bearish_1d = close_1d < ema_50_1d
    
    # Calculate ATR(14) on 1d
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    # Use rolling window with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_atr = 0.0  # store ATR at entry for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR volatility filter: only trade when ATR < 3% of price (low volatility regime)
        vol_filter = atr_14_aligned[i] < 0.03 * close[i]
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 1d bullish trend AND low volatility
            if (close[i] > donchian_upper[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                vol_filter):
                signals[i] = 0.25
                position = 1
                entry_atr = atr_14_aligned[i]
            # Short conditions: price breaks below Donchian lower AND 1d bearish trend AND low volatility
            elif (close[i] < donchian_lower[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  vol_filter):
                signals[i] = -0.25
                position = -1
                entry_atr = atr_14_aligned[i]
        elif position == 1:
            # Exit long: price closes below Donchian lower OR 1d trend turns bearish OR stoploss hit
            stoploss_level = close[i - 1] - 2.5 * entry_atr if i > 0 else close[i] - 2.5 * atr_14_aligned[i]
            if (close[i] < donchian_lower[i] or 
                trend_bearish_aligned[i] > 0.5 or
                close[i] < stoploss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR 1d trend turns bullish OR stoploss hit
            stoploss_level = close[i - 1] + 2.5 * entry_atr if i > 0 else close[i] + 2.5 * atr_14_aligned[i]
            if (close[i] > donchian_upper[i] or 
                trend_bullish_aligned[i] > 0.5 or
                close[i] > stoploss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals