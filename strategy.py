#!/usr/bin/env python3
name = "6h_OrderBlock_OrderFlow_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== Order Block Detection (6h) =====
    # Bullish OB: bearish candle followed by bullish breakout
    # Bearish OB: bullish candle followed by bearish breakdown
    ob_bull = np.zeros(n, dtype=bool)
    ob_bear = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # Bullish OB: red candle (close < open) then bullish breakout (close > high of red candle)
        if (close[i-2] < prices['open'].iloc[i-2] and  # red candle
            close[i] > high[i-2]):  # breaks above red candle's high
            ob_bull[i] = True
        # Bearish OB: green candle (close > open) then bearish breakdown (close < low of green candle)
        elif (close[i-2] > prices['open'].iloc[i-2] and  # green candle
              close[i] < low[i-2]):  # breaks below green candle's low
            ob_bear[i] = True
    
    # ===== Weekly Trend Filter =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ===== Daily Volume Spike Filter =====
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish OB + above weekly EMA50 + daily volume spike
            if (ob_bull[i] and
                close[i] > ema50_1w_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Bearish OB + below weekly EMA50 + daily volume spike
            elif (ob_bear[i] and
                  close[i] < ema50_1w_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish OB formed or closes below weekly EMA50
            if ob_bear[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish OB formed or closes above weekly EMA50
            if ob_bull[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals