#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Pullback_Strategy
Hypothesis: Intraday pullbacks to Camarilla pivot levels (P, R3, S3) on 4h chart with daily context provide high-probability entries.
Buy near S3/S4 in daily uptrend, sell near R3/R4 in daily downtrend. Uses volume confirmation and avoids choppy markets.
Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend) markets. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Pullback_Strategy"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CONTEXT ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily trend: close vs 20-period EMA
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_uptrend = close_1d > ema20_1d
    daily_downtrend = close_1d < ema20_1d
    
    # Calculate Camarilla levels for each day
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    r3 = close_1d + rang * 1.1 / 4.0
    r4 = close_1d + rang * 1.1 / 2.0
    s3 = close_1d - rang * 1.1 / 4.0
    s4 = close_1d - rang * 1.1 / 2.0
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # === VOLUME CONFIRMATION (4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === CHOPPINESS FILTER (4h) ===
    def pine_atr(high, low, close, length):
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False, min_periods=1).mean().values
        return atr
    
    atr14 = pine_atr(high, low, close, 14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=1).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=1).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (atr14 * 14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Avoid choppy markets
        if chop[i] > 61.8:  # choppy
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: pullback to S3/S4 in daily uptrend with volume
        long_setup = (daily_uptrend_aligned[i] > 0.5) and \
                     (low[i] <= s3_aligned[i]) and \
                     (close[i] > open_[i] if 'open_' in locals() else True) and \
                     (vol_ratio[i] > 1.3)
        # Simplify: use close > prior close for bullish candle
        if i > 0:
            bullish_candle = close[i] > close[i-1]
        else:
            bullish_candle = True
        long_setup = (daily_uptrend_aligned[i] > 0.5) and \
                     (low[i] <= s3_aligned[i]) and \
                     bullish_candle and \
                     (vol_ratio[i] > 1.3)
        
        # Short: rally to R3/R4 in daily downtrend with volume
        if i > 0:
            bearish_candle = close[i] < close[i-1]
        else:
            bearish_candle = True
        short_setup = (daily_downtrend_aligned[i] > 0.5) and \
                      (high[i] >= r3_aligned[i]) and \
                      bearish_candle and \
                      (vol_ratio[i] > 1.3)
        
        # Exit conditions
        exit_long = position == 1 and (close[i] >= r3_aligned[i] or chop[i] > 61.8)
        exit_short = position == -1 and (close[i] <= s3_aligned[i] or chop[i] > 61.8)
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals