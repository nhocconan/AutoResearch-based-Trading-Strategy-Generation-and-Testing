#!/usr/bin/env python3
"""
Hypothesis: 1h EMA crossover with 4h trend filter and 1d volatility regime.
Long when 1h EMA(9) crosses above EMA(21), 4h close > EMA(50), and 1d ATR ratio < 0.8.
Short when 1h EMA(9) crosses below EMA(21), 4h close < EMA(50), and 1d ATR ratio < 0.8.
Exit when opposite crossover occurs or 1d ATR ratio > 1.2.
Uses 4h for trend direction, 1d for volatility filter, 1h for entry timing.
Designed to capture trends in low volatility environments which work in both bull and bear markets.
Target: 20-40 trades/year per symbol to minimize fee drag on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1h EMA(9) and EMA(21)
    close_s = pd.Series(close)
    ema9 = close_s.ewm(span=9, min_periods=9, adjust=False).mean().values
    ema21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate 4h EMA(50) for trend filter
    close_4h_s = pd.Series(close_4h)
    ema50_4h = close_4h_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d ATR (14-period) and ATR ratio (current ATR / 20-period average ATR)
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_1d / atr_ma_20_1d, np.nan)
    
    # Align 1d ATR ratio to 1h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema9[i]) or 
            np.isnan(ema21[i]) or
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(atr_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h close > EMA(50) for long, < EMA(50) for short
        uptrend_4h = close_4h[-1] > ema50_4h[-1] if len(close_4h) > 0 else False  # Simplified - using last value for demo
        downtrend_4h = close_4h[-1] < ema50_4h[-1] if len(close_4h) > 0 else False
        # Proper aligned trend check
        uptrend_4h = not np.isnan(ema50_4h_aligned[i]) and close[i] > ema50_4h_aligned[i]
        downtrend_4h = not np.isnan(ema50_4h_aligned[i]) and close[i] < ema50_4h_aligned[i]
        
        # Volatility regime filter: 1d ATR ratio < 0.8 (low volatility environment)
        vol_regime = not np.isnan(atr_ratio_1d_aligned[i]) and atr_ratio_1d_aligned[i] < 0.8
        
        # EMA crossover signals
        ema_cross_up = ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]
        ema_cross_down = ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]
        
        # Exit conditions
        exit_long = ema9[i] < ema21[i]  # opposite crossover
        exit_short = ema9[i] > ema21[i]  # opposite crossover
        high_vol_exit = not np.isnan(atr_ratio_1d_aligned[i]) and atr_ratio_1d_aligned[i] > 1.2
        
        if position == 0:
            # Long: EMA cross up with 4h uptrend and low volatility regime
            if (ema_cross_up and uptrend_4h and vol_regime):
                signals[i] = 0.20
                position = 1
            # Short: EMA cross down with 4h downtrend and low volatility regime
            elif (ema_cross_down and downtrend_4h and vol_regime):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: EMA cross down OR high volatility environment
            if (exit_long or high_vol_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: EMA cross up OR high volatility environment
            if (exit_short or high_vol_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA9_21_Crossover_4hTrend_1dATRRegime"
timeframe = "1h"
leverage = 1.0