#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h trend filter (EMA50) and 1d momentum filter (ROC > 0)
# Long when price > 4h EMA50 AND 1d ROC > 0 AND price breaks above 1h Donchian upper (10-period)
# Short when price < 4h EMA50 AND 1d ROC < 0 AND price breaks below 1h Donchian lower (10-period)
# Exit when price crosses 1h Donchian midline
# Uses 4h for trend direction, 1d for momentum filter, 1h for entry timing
# Target: 60-150 total trades over 4 years (15-37/year) with strict filters to avoid overtrading

name = "1h_ema50_roc_donchian10_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h Donchian Channel (10-period)
    highest_high = pd.Series(high).rolling(window=10, min_periods=10).max()
    lowest_low = pd.Series(low).rolling(window=10, min_periods=10).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 4h EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_4h = close_4h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ROC(1) momentum filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    roc_1d = np.zeros_like(close_1d)
    roc_1d[1:] = (close_1d[1:] - close_1d[:-1]) / close_1d[:-1] * 100
    roc_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_1d)
    
    # Session filter: 8-20 UTC (reduce noise trades)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(roc_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with strict filters
            # Long: price > 4h EMA50 AND 1d ROC > 0 AND break above Donchian upper
            if (close[i] > ema_4h_aligned[i] and roc_1d_aligned[i] > 0 and
                close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]):
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA50 AND 1d ROC < 0 AND break below Donchian lower
            elif (close[i] < ema_4h_aligned[i] and roc_1d_aligned[i] < 0 and
                  close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]):
                signals[i] = -0.20
                position = -1
    
    return signals