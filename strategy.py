#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with weekly trend filter.
# Uses 20-period BB with 2 std dev and Bollinger Width percentile to detect low volatility squeezes.
# Breakout occurs when price closes outside BB after a squeeze (BW < 20th percentile).
# Weekly trend filter (price > weekly EMA50) ensures trades align with higher timeframe momentum.
# Works in bull markets (breakouts with trend) and bear markets (mean reversion fails, trend filter avoids false breakouts).
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6h_bb_squeeze_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    
    # Bollinger Width (normalized by middle band)
    bw = (upper - lower) / ma
    
    # Bollinger Width percentile (20-period lookback) to detect squeeze
    bw_percentile = pd.Series(bw).rolling(window=20, min_periods=20).rank(pct=True) * 100
    
    # Weekly trend filter: price > weekly EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_ema = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(bw_percentile[i]) or np.isnan(weekly_ema_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Squeeze condition: Bollinger Width below 20th percentile (low volatility)
        squeeze = bw_percentile[i] < 20
        
        if position == 1:  # long position
            # Exit: price reaches middle band (mean reversion) or weekly trend fails
            if close[i] <= ma[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches middle band or weekly trend fails
            if close[i] >= ma[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries after squeeze
            if squeeze:
                # Bullish breakout: close above upper band with weekly uptrend
                if close[i] > upper[i] and close[i] > weekly_ema_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: close below lower band with weekly downtrend
                elif close[i] < lower[i] and close[i] < weekly_ema_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals