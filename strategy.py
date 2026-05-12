#!/usr/bin/env python3
# 12h_RVI_Crossover_1dTrendFilter
# Hypothesis: The Relative Vigor Index (RVI) identifies momentum shifts. On 12h timeframe,
# enter long when RVI crosses above its signal line with 1d EMA50 uptrend (close > EMA50).
# Enter short when RVI crosses below its signal line with 1d EMA50 downtrend (close < EMA50).
# Exit when RVI crosses back (mean reversion). Uses 1d trend filter to avoid counter-trend trades.
# Targets 20-40 trades/year for low fee drag and works in both bull and bear markets.

name = "12h_RVI_Crossover_1dTrendFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate RVI (Relative Vigor Index) on 12h data
    # RVI = (Close - Open) / (High - Low) smoothed
    numerator = close - open_
    denominator = high - low
    # Avoid division by zero
    rvi_raw = np.where(denominator != 0, numerator / denominator, 0)
    
    # Smooth RVI with 10-period SMA (standard)
    rvi_numerator = pd.Series(rvi_raw).rolling(window=10, min_periods=10).mean().values
    rvi_denominator = pd.Series(np.where(denominator != 0, 1.0, 0)).rolling(window=10, min_periods=10).mean().values
    rvi = np.where(rvi_denominator != 0, rvi_numerator / rvi_denominator, 0)
    
    # Signal line: 4-period SMA of RVI
    rvi_signal = pd.Series(rvi).rolling(window=4, min_periods=4).mean().values
    
    # Smooth the inputs for final RVI calculation (standard method)
    # Actually, standard RVI uses SMA of numerator and denominator separately
    num_smooth = pd.Series(close - open_).rolling(window=10, min_periods=10).mean().values
    den_smooth = pd.Series(high - low).rolling(window=10, min_periods=10).mean().values
    rvi_final = np.where(den_smooth != 0, num_smooth / den_smooth, 0)
    rvi_signal_final = pd.Series(rvi_final).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(rvi_final[i]) or np.isnan(rvi_signal_final[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        rvi_val = rvi_final[i]
        rvi_sig_val = rvi_signal_final[i]
        ema1d_trend = ema50_1d_aligned[i]
        
        if position == 0:
            # LONG: RVI crosses above signal line with 1d uptrend
            if rvi_val > rvi_sig_val and rvi_final[i-1] <= rvi_signal_final[i-1] and close[i] > ema1d_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: RVI crosses below signal line with 1d downtrend
            elif rvi_val < rvi_sig_val and rvi_final[i-1] >= rvi_signal_final[i-1] and close[i] < ema1d_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RVI crosses below signal line (mean reversion)
            if rvi_val < rvi_sig_val and rvi_final[i-1] >= rvi_signal_final[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RVI crosses above signal line (mean reversion)
            if rvi_val > rvi_sig_val and rvi_final[i-1] <= rvi_signal_final[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals