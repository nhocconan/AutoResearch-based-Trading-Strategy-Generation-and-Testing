#!/usr/bin/env python3
# 12H_MACD_CROSSOVER_1D_TREND_FILTER
# Hypothesis: MACD crossover (12,26,9) on 12h timeframe captures momentum swings. 
# Trend filter uses 1d EMA50 to ensure trades align with higher timeframe direction, 
# avoiding counter-trend trades in both bull and bear markets. 
# Volume confirmation filters low-activity breakouts. 
# Target: 20-30 trades/year on 12h timeframe with strict entry conditions.

name = "12H_MACD_CROSSOVER_1D_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period volume average for confirmation
    vol_ma20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # MACD on 12h close prices
    close_series = pd.Series(close)
    ema12 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close_series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd - signal_line
    
    macd_values = macd.values
    signal_values = signal_line.values
    macd_hist_values = macd_hist.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(macd_values[i]) or np.isnan(signal_values[i]) or np.isnan(macd_hist_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + MACD bullish crossover + volume confirmation
            if (close[i] > ema50_1d_aligned[i] and
                macd_values[i] > signal_values[i] and
                macd_values[i-1] <= signal_values[i-1] and
                volume[i] > vol_ma20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + MACD bearish crossover + volume confirmation
            elif (close[i] < ema50_1d_aligned[i] and
                  macd_values[i] < signal_values[i] and
                  macd_values[i-1] >= signal_values[i-1] and
                  volume[i] > vol_ma20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or MACD bearish crossover
            if (close[i] <= ema50_1d_aligned[i] or
                macd_values[i] < signal_values[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or MACD bullish crossover
            if (close[i] >= ema50_1d_aligned[i] or
                macd_values[i] > signal_values[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals