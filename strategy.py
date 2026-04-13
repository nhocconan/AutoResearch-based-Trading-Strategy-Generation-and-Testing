#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ADX trend strength combined with 1-week ATR-based breakout for trend following.
# Uses weekly ATR to set breakout levels and daily ADX to confirm trend strength.
# This strategy aims to capture strong trends while avoiding choppy markets.
# Weekly timeframe reduces noise and false breakouts, while daily ADX ensures we only trade in strong trends.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ATR for volatility measurement
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) calculation
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly breakout levels: ±1.5 * ATR from weekly open
    open_1w = df_1w['open'].values
    upper_breakout = open_1w + (1.5 * atr_1w)
    lower_breakout = open_1w - (1.5 * atr_1w)
    
    # Daily data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range for daily
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_d[0] = 0
    tr2_d[0] = 0
    tr3_d[0] = 0
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    
    # Smoothed values
    atr_1d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_1d)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    
    # Align all data to daily timeframe
    upper_breakout_aligned = align_htf_to_ltf(prices, df_1w, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_1w, lower_breakout)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_breakout_aligned[i]) or np.isnan(lower_breakout_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ADX condition: trend strength > 25
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions: Weekly breakout with ADX trend filter
        breakout_long = close[i] > upper_breakout_aligned[i]
        breakout_short = close[i] < lower_breakout_aligned[i]
        
        if position == 0:
            if breakout_long and strong_trend:
                position = 1
                signals[i] = position_size
            elif breakout_short and strong_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below weekly lower breakout or ADX weakens
            if close[i] < lower_breakout_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above weekly upper breakout or ADX weakens
            if close[i] > upper_breakout_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_ADX_Trend_Breakout_ATR_v1"
timeframe = "1d"
leverage = 1.0