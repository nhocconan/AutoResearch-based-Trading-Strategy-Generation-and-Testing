#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 4h ADX filter and 1d trend bias
# Uses 4h ADX > 25 to filter for trending markets only
# Uses 1d close > SMA50 for long bias, < SMA50 for short bias
# Entry on 1h when price crosses above/below 20-period EMA with momentum
# Designed for 15-30 trades/year to avoid fee drag
name = "1h_ADX_Trend_EMA_Cross"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h ADX for trend strength filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX components
    plus_dm = np.diff(df_4h['high'].values)
    minus_dm = np.diff(df_4h['low'].values)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr = np.maximum(np.diff(df_4h['high'].values), np.diff(df_4h['low'].values))
    tr = np.maximum(tr, np.abs(np.diff(df_4h['close'].values)))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / tr_smooth
    minus_di = 100 * wilders_smoothing(minus_dm, period) / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 1d trend bias: close vs SMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    sma_50_1d = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 1h EMA20 for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > EMA20 and above 1d SMA50 (bullish bias)
            if close[i] > ema_20[i] and close[i] > sma_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price < EMA20 and below 1d SMA50 (bearish bias)
            elif close[i] < ema_20[i] and close[i] < sma_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price < EMA20 or trend weakens (ADX < 20)
            if close[i] < ema_20[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price > EMA20 or trend weakens (ADX < 20)
            if close[i] > ema_20[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals