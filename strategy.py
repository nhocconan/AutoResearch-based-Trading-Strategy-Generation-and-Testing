#!/usr/bin/env python3
# Hypothesis: 6h timeframe with 1-week ADX trend filter and 1-day RSI mean reversion.
# In high ADX (>25) trending markets, follow 1-day RSI momentum (RSI>55 long, RSI<45 short).
# In low ADX (<20) ranging markets, fade extreme RSI (RSI>70 short, RSI<30 long).
# Uses 1-week ADX for regime detection and 1-day RSI for signals, avoiding whipsaws in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_WeeklyADX_DailyRSI_Regime"
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
    volume = prices['volume'].values
    
    # Calculate 1-week ADX(14) for trend regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # True Range and Directional Movement
    prev_close = np.roll(df_1w['close'], 1)
    prev_high = np.roll(df_1w['high'], 1)
    prev_low = np.roll(df_1w['low'], 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - prev_close)
    tr3 = np.abs(df_1w['low'] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = df_1w['high'] - prev_high
    down_move = prev_low - df_1w['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Regime detection
    adx_high = adx_1w > 25   # Trending regime
    adx_low = adx_1w < 20    # Ranging regime
    adx_high_aligned = align_htf_to_ltf(prices, df_1w, adx_high)
    adx_low_aligned = align_htf_to_ltf(prices, df_1w, adx_low)
    
    # Calculate 1-day RSI(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    delta = np.diff(df_1d['close'], prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_high_aligned[i]) or np.isnan(adx_low_aligned[i]) or
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: trending + RSI>55 OR ranging + RSI<30
            if (adx_high_aligned[i] and rsi_1d_aligned[i] > 55) or \
               (adx_low_aligned[i] and rsi_1d_aligned[i] < 30):
                signals[i] = 0.25
                position = 1
            # Enter short: trending + RSI<45 OR ranging + RSI>70
            elif (adx_high_aligned[i] and rsi_1d_aligned[i] < 45) or \
                 (adx_low_aligned[i] and rsi_1d_aligned[i] > 70):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakens OR RSI reverses
            if not adx_high_aligned[i] or rsi_1d_aligned[i] < 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakens OR RSI reverses
            if not adx_high_aligned[i] or rsi_1d_aligned[i] > 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals