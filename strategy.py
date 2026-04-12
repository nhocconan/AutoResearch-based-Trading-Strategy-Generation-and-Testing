#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Pullback_v1
Hypothesis: Use 1d trend via EMA200 and 4h for pullback entries at 4h EMA21 during pullbacks.
Long when 1d EMA200 up and price pulls back to 4h EMA21 with RSI<40.
Short when 1d EMA200 down and price pulls back to 4h EMA21 with RSI>60.
Only trade during 08-20 UTC to avoid low liquidity periods.
Target: 20-40 trades per year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Get 4H data for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # === 1D TREND: EMA200 ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4H ENTRY: EMA21 ===
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # === 1H RSI (14) ===
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(200, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data invalid
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema21_4h_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Entry conditions
        uptrend = close[i] > ema200_1d_aligned[i]
        downtrend = close[i] < ema200_1d_aligned[i]
        pullback_to_ema = abs(close[i] - ema21_4h_aligned[i]) / ema21_4h_aligned[i] < 0.005  # 0.5% tolerance
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        long_entry = uptrend and pullback_to_ema and rsi_oversold
        short_entry = downtrend and pullback_to_ema and rsi_overbought
        
        # Exit conditions: reverse signal or RSI normalization
        long_exit = (not uptrend) or rsi[i] > 50
        short_exit = (not downtrend) or rsi[i] < 50
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals