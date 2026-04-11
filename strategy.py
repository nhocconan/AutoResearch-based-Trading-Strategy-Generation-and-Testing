#!/usr/bin/env python3
# 6h_1w_1d_vwap_mean_reversion_v1
# Strategy: 6h VWAP mean reversion with 1w/1d trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Price tends to revert to VWAP during ranging markets. Strong weekly and daily trends filter out false signals. Works in bull/bear by only taking mean-reversion trades in the direction of the higher timeframe trend, reducing whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_vwap_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, np.nan)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w EMA20 for trend filter (slower)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if np.isnan(vwap[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filters: price above/below EMA
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        uptrend_1w = close[i] > ema_20_1w_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        downtrend_1w = close[i] < ema_20_1w_aligned[i]
        
        # Deviation from VWAP
        dev_pct = (close[i] - vwap[i]) / vwap[i] if vwap[i] != 0 else 0
        
        # Entry conditions: mean reversion in direction of trend
        # Long: Price below VWAP AND both timeframes uptrend
        if dev_pct < -0.008 and uptrend_1d and uptrend_1w and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price above VWAP AND both timeframes downtrend
        elif dev_pct > 0.008 and downtrend_1d and downtrend_1w and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses VWAP (mean reversion complete)
        elif position == 1 and close[i] > vwap[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < vwap[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals