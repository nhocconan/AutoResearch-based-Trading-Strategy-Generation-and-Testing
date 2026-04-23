#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w trend filter and ATR-based volatility filter.
Long when price breaks above upper Donchian band AND weekly close > weekly EMA34 AND ATR(14) > ATR(50) (expanding volatility).
Short when price breaks below lower Donchian band AND weekly close < weekly EMA34 AND ATR(14) > ATR(50).
Exit when price crosses the 10-period EMA (mean reversion signal).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Donchian breakouts capture strong momentum moves, weekly trend filter avoids counter-trend trades,
volatility expansion filter ensures breakouts occur during genuine momentum bursts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate ATR(14) and ATR(50) on 1d data for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(np.roll(close_1d, 1) - high_1d)
    low_close = np.abs(np.roll(close_1d, 1) - low_1d)
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    # Set first value to high_low to avoid NaN from roll
    true_range[0] = high_low[0]
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(true_range).rolling(window=50, min_periods=50).mean().values
    
    # Calculate EMA34 on 1w data for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian bands (20-period) on 1d data
    upper_donchian = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period EMA on 1d data for exit signal
    ema10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align all 1d indicators to 1d timeframe (self-alignment for proper indexing)
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1d, lower_donchian)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    ema10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema10_1d)
    
    # Align 1w indicators to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or
            np.isnan(ema10_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend_up = close_1d[i] > ema34_1w_aligned[i]  # using 1d close vs weekly EMA
        weekly_trend_down = close_1d[i] < ema34_1w_aligned[i]
        volatility_expanding = atr_14_aligned[i] > atr_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian AND weekly uptrend AND volatility expanding
            if (close_1d[i] > upper_donchian_aligned[i] and weekly_trend_up and 
                volatility_expanding):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian AND weekly downtrend AND volatility expanding
            elif (close_1d[i] < lower_donchian_aligned[i] and weekly_trend_down and 
                  volatility_expanding):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 10-period EMA (mean reversion signal)
                if close_1d[i] < ema10_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 10-period EMA (mean reversion signal)
                if close_1d[i] > ema10_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA34_VolatilityFilter"
timeframe = "1d"
leverage = 1.0