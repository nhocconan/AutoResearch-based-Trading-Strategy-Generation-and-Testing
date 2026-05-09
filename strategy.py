#!/usr/bin/env python3
# Hypothesis: 6h timeframe with 1-day Supertrend (ATR=10, multiplier=3) as trend filter and 1-day RSI(14) for mean-reversion entries.
# In uptrend (price above Supertrend), go long when RSI crosses below 30 (oversold pullback).
# In downtrend (price below Supertrend), go short when RSI crosses above 70 (overbought bounce).
# Exits when price crosses Supertrend in opposite direction or RSI reaches opposite extreme (70 for long, 30 for short).
# Supertrend provides objective trend direction that works in both bull and bear markets.
# RSI extremes offer high-probability mean-reversion entries within the trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_Supertrend_RSI_MeanReversion"
timeframe = "6h"
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
    
    # Calculate 1-day Supertrend (ATR=10, multiplier=3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # True Range
    prev_close = np.roll(df_1d['close'], 1)
    prev_close[0] = np.nan
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - prev_close)
    tr3 = np.abs(df_1d['low'] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (df_1d['high'] + df_1d['low']) / 2.0
    upper_basic = hl2 + 3.0 * atr
    lower_basic = hl2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    upper = np.full_like(upper_basic, np.nan)
    lower = np.full_like(lower_basic, np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            upper[i] = upper_basic[i]
            lower[i] = lower_basic[i]
        else:
            if upper_basic[i] <= upper[i-1] or df_1d['close'].iloc[i-1] > upper[i-1]:
                upper[i] = upper_basic[i]
            else:
                upper[i] = upper[i-1]
            
            if lower_basic[i] >= lower[i-1] or df_1d['close'].iloc[i-1] < lower[i-1]:
                lower[i] = lower_basic[i]
            else:
                lower[i] = lower[i-1]
    
    # Supertrend
    supertrend = np.full_like(close, np.nan)
    for i in range(len(df_1d)):
        if i == 0:
            supertrend[i] = upper_basic[i]
        else:
            if supertrend[i-1] == upper[i-1]:
                if df_1d['close'].iloc[i] <= upper[i]:
                    supertrend[i] = upper[i]
                else:
                    supertrend[i] = lower[i]
            else:
                if df_1d['close'].iloc[i] >= lower[i]:
                    supertrend[i] = lower[i]
                else:
                    supertrend[i] = upper[i]
    
    # Supertrend trend direction: 1 = uptrend (price above supertrend), -1 = downtrend
    trend_dir = np.where(close > supertrend, 1, -1)
    
    # Calculate 1-day RSI(14)
    delta = pd.Series(df_1d['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to LTF
    trend_dir_aligned = align_htf_to_ltf(prices, df_1d, trend_dir)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # RSI crossover signals
    rsi_prev = np.roll(rsi_aligned, 1)
    rsi_prev[0] = 50  # neutral start
    
    rsi_cross_below_30 = (rsi_prev > 30) & (rsi_aligned <= 30)
    rsi_cross_above_70 = (rsi_prev < 70) & (rsi_aligned >= 70)
    rsi_cross_below_70 = (rsi_prev > 70) & (rsi_aligned <= 70)
    rsi_cross_above_30 = (rsi_prev < 30) & (rsi_aligned >= 30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_dir_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(rsi_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: uptrend + RSI crosses below 30 (oversold pullback)
            if trend_dir_aligned[i] == 1 and rsi_cross_below_30[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + RSI crosses above 70 (overbought bounce)
            elif trend_dir_aligned[i] == -1 and rsi_cross_above_70[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: downtrend OR RSI crosses above 70 (overbought)
            if trend_dir_aligned[i] == -1 or rsi_cross_above_70[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: uptrend OR RSI crosses below 30 (oversold)
            if trend_dir_aligned[i] == 1 or rsi_cross_below_30[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals