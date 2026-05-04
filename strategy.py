#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and session filter (08-20 UTC)
# Long when RSI < 30 AND 4h close > 4h EMA50 AND session active (08-20 UTC)
# Short when RSI > 70 AND 4h close < 4h EMA50 AND session active (08-20 UTC)
# Exit when RSI crosses back to neutral (40 for longs, 60 for shorts) or trend reverses
# Uses 4h EMA50 for major trend filter to avoid counter-trend trades, targeting 15-35 trades/year on 1h.
# Session filter reduces noise trades during low-volume off-hours. RSI extremes provide mean reversion edge.
# Works in bull markets via selective longs in bullish 4h trend regime and bear markets via shorts in bearish 4h trend regime.

name = "1h_RSI14_4hTrend_SessionFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Session filter: 08-20 UTC (pre-compute before loop)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_active = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_4h = close_4h > ema_50_4h
    trend_bearish_4h = close_4h < ema_50_4h
    
    # Align 4h trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish_4h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish_4h.astype(float))
    
    # Calculate RSI(14) from 1h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(rsi[i]) or not session_active[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI < 30 AND 4h bullish trend AND session active
            if (rsi[i] < 30 and 
                trend_bullish_aligned[i] > 0.5):  # 4h bullish trend
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI > 70 AND 4h bearish trend AND session active
            elif (rsi[i] > 70 and 
                  trend_bearish_aligned[i] > 0.5):  # 4h bearish trend
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 40 OR 4h trend turns bearish
            if (rsi[i] > 40 or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI crosses below 60 OR 4h trend turns bullish
            if (rsi[i] < 60 or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals