#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_pullback_strategy_v1
# Buy pullbacks to 4h EMA20 in 1d uptrend, sell rallies to 4h EMA20 in 1d downtrend.
# Uses 4h for trend direction and dynamic support/resistance, 1d for higher-timeframe trend filter,
# and 1h for precise entry timing. Low-frequency trades expected due to multiple confluence filters.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
name = "1h_4h_1d_pullback_strategy_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for EMA20 and ATR (dynamic support/resistance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for higher-timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA20
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h ATR for dynamic thresholds
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    tr = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - close_4h[:-1]),
            np.abs(low_4h[1:] - close_4h[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            not (8 <= hours[i] <= 20)):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Calculate dynamic bands around 4h EMA20
        upper_band = ema_20_4h_aligned[i] + 0.5 * atr_4h_aligned[i]
        lower_band = ema_20_4h_aligned[i] - 0.5 * atr_4h_aligned[i]
        
        # Determine 1d trend
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Long setup: price pulls back to lower band in 1d uptrend
        long_setup = (
            uptrend_1d and
            low[i] <= lower_band and  # touched or went below support
            close[i] > ema_20_4h_aligned[i]  # closing back above EMA20
        )
        
        # Short setup: price rallies to upper band in 1d downtrend
        short_setup = (
            downtrend_1d and
            high[i] >= upper_band and  # touched or went above resistance
            close[i] < ema_20_4h_aligned[i]  # closing back below EMA20
        )
        
        # Exit conditions
        exit_long = (
            close[i] >= ema_20_4h_aligned[i] + atr_4h_aligned[i] or  # reached upper band
            (position == 1 and not uptrend_1d)  # 1d trend turned down
        )
        
        exit_short = (
            close[i] <= ema_20_4h_aligned[i] - atr_4h_aligned[i] or  # reached lower band
            (position == -1 and not downtrend_1d)  # 1d trend turned up
        )
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals