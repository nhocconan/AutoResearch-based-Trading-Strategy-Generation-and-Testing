#!/usr/bin/env python3
# 4h_CCI_Trend_Filter_12hEMA50
# Hypothesis: Uses CCI(20) on 4h to detect momentum extremes and 12h EMA(50) as trend filter.
# Enters long when CCI crosses above -100 (bullish momentum) with price above 12h EMA50.
# Enters short when CCI crosses below +100 (bearish momentum) with price below 12h EMA50.
# Exits when CCI returns to neutral zone (-100 to +100) or trend filter fails.
# Designed for 20-40 trades/year on 4h with strong trend persistence in both bull and bear markets.

name = "4h_CCI_Trend_Filter_12hEMA50"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # CCI(20) calculation: (Typical Price - SMA(TP,20)) / (0.015 * Mean Deviation)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    
    sma_tp_20 = tp_series.rolling(window=20, min_periods=20).mean()
    mean_deviation = tp_series.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    )
    
    # Avoid division by zero
    cci = np.where(mean_deviation != 0, (tp_series - sma_tp_20) / (0.015 * mean_deviation), 0)
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure sufficient warmup for CCI and EMA
    
    for i in range(start_idx, n):
        if np.isnan(cci[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # CCI signals: long when crosses above -100, short when crosses below +100
        cci_long_signal = cci[i] > -100 and (i == start_idx or cci[i-1] <= -100)
        cci_short_signal = cci[i] < 100 and (i == start_idx or cci[i-1] >= 100)
        
        if position == 0:
            # Long: CCI bullish momentum + price above 12h EMA50 (uptrend)
            if cci_long_signal and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: CCI bearish momentum + price below 12h EMA50 (downtrend)
            elif cci_short_signal and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: CCI returns to neutral or trend fails
            if cci[i] >= -100 and cci[i] <= 100 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: CCI returns to neutral or trend fails
            if cci[i] >= -100 and cci[i] <= 100 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals