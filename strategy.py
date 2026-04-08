#!/usr/bin/env python3
# 1d_1w_keltner_squeeze_breakout_v1
# Hypothesis: Daily Keltner channel breakout with weekly trend filter to capture strong trends while avoiding whipsaws.
# Long when price breaks above upper Keltner band and weekly EMA21 > weekly EMA50.
# Short when price breaks below lower Keltner band and weekly EMA21 < weekly EMA50.
# Exit when price re-enters Keltner channel or weekly trend reverses.
# Uses Keltner channels on daily for volatility-based breakouts and weekly EMA for trend filter.
# Designed to generate ~10-20 trades/year to minimize fee decay while capturing major trends.

import numpy as np
import pandas as pd
from mfe_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_squeeze_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ATR for Keltner channels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(1, n):
        if i < 14:
            atr[i] = np.nan
        else:
            atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Keltner channels (20-period EMA +/- 2*ATR)
    ema20 = np.full(n, np.nan)
    ema20_smooth = close.astype(float)
    for i in range(n):
        if i == 0:
            ema20[i] = close[i]
        else:
            ema20[i] = 0.1 * close[i] + 0.9 * ema20[i-1]
    
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA21 and EMA50 for trend filter
    ema21_1w = np.full(len(close_1w), np.nan)
    ema50_1w = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i == 0:
            ema21_1w[i] = close_1w[i]
            ema50_1w[i] = close_1w[i]
        else:
            ema21_1w[i] = 0.09 * close_1w[i] + 0.91 * ema21_1w[i-1]
            ema50_1w[i] = 0.04 * close_1w[i] + 0.96 * ema50_1w[i-1]
    
    # Align weekly EMAs to daily timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        upper_keltner_val = upper_keltner[i]
        lower_keltner_val = lower_keltner[i]
        ema21_1w_val = ema21_1w_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price re-enters Keltner channel or weekly trend reverses (EMA21 < EMA50)
            if price <= upper_keltner_val and price >= lower_keltner_val:
                position = 0
                signals[i] = 0.0
            elif ema21_1w_val < ema50_1w_val:  # Bearish trend reversal
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price re-enters Keltner channel or weekly trend reverses (EMA21 > EMA50)
            if price <= upper_keltner_val and price >= lower_keltner_val:
                position = 0
                signals[i] = 0.0
            elif ema21_1w_val > ema50_1w_val:  # Bullish trend reversal
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Keltner breakout with weekly trend alignment
            # Bullish: price breaks above upper Keltner and weekly EMA21 > EMA50
            if price > upper_keltner_val and ema21_1w_val > ema50_1w_val:
                position = 1
                signals[i] = 0.25
            # Bearish: price breaks below lower Keltner and weekly EMA21 < EMA50
            elif price < lower_keltner_val and ema21_1w_val < ema50_1w_val:
                position = -1
                signals[i] = -0.25
    
    return signals