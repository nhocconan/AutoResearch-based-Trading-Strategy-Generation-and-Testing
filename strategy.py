#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day Keltner Channel squeeze and 1-week EMA trend filter.
# In low volatility regimes (Keltner width < 20th percentile), price tends to mean-revert to the 20 EMA.
# Enters long when price crosses above 20 EMA in low-volatility regime and price > 1w EMA50 (bullish trend).
# Enters short when price crosses below 20 EMA in low-volatility regime and price < 1w EMA50 (bearish trend).
# Uses 1-day Keltner Channel squeeze as volatility filter and 1-week EMA50 as trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_Keltner_Squeeze_WeeklyEMA_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Keltner Channel (20 EMA, ATRx2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    
    # EMA20
    ema_20 = close_1d.ewm(span=20, adjust=False, min_periods=20).mean()
    # ATR
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False, min_periods=20).mean()
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    keltner_width = upper_keltner - lower_keltner
    
    # Keltner Channel squeeze: low volatility when width < 20th percentile
    keltner_width_percentile = keltner_width.rolling(window=100, min_periods=100).quantile(0.2)
    keltner_squeeze = keltner_width < keltner_width_percentile
    keltner_squeeze_values = keltner_squeeze.values
    keltner_squeeze_aligned = align_htf_to_ltf(prices, df_1d, keltner_squeeze_values)
    
    # EMA20 for mean reversion target
    ema_20_values = ema_20.values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_values)
    
    # Price position relative to EMA20
    price_above_ema = close > ema_20_aligned
    price_below_ema = close < ema_20_aligned
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1w_values = ema_50_1w.values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_values)
    
    # Trend condition: price > 1w EMA50 for long, price < 1w EMA50 for short
    price_above_weekly_ema = close > ema_50_1w_aligned
    price_below_weekly_ema = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(keltner_squeeze_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or
            np.isnan(price_above_ema[i]) or np.isnan(price_below_ema[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(price_above_weekly_ema[i]) or np.isnan(price_below_weekly_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: low volatility (Keltner squeeze) + price above EMA20 + price > 1w EMA50
            if keltner_squeeze_aligned[i] and price_above_ema[i] and price_above_weekly_ema[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: low volatility (Keltner squeeze) + price below EMA20 + price < 1w EMA50
            elif keltner_squeeze_aligned[i] and price_below_ema[i] and price_below_weekly_ema[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility regime shifts to high OR price crosses below EMA20
            if (not keltner_squeeze_aligned[i]) or (not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility regime shifts to high OR price crosses above EMA20
            if (not keltner_squeeze_aligned[i]) or (not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals