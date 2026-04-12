#!/usr/bin/env python3
"""
4h_1d_Camarilla_Volume_Momentum_Strategy
Hypothesis: Breakouts above daily Camarilla R4 with volume momentum and 4h momentum filter
capture institutional breakout moves. Works in bull markets via breakouts and bear markets
via breakdowns below S4. Uses 4h RSI for momentum confirmation to avoid false breakouts.
Target: 20-40 trades/year to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Volume_Momentum_Strategy"
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
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R4 = Close + Range * 1.1/2
    # S4 = Close - Range * 1.1/2
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    r4 = close_1d + rang * 1.1 / 2.0
    s4 = close_1d - rang * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 4H MOMENTUM FILTER ===
    # RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout above R4 with volume momentum and RSI > 50
        long_breakout = (close[i] > r4_aligned[i]) and (vol_ratio[i] > 1.5) and (rsi[i] > 50)
        
        # Breakdown below S4 with volume momentum and RSI < 50
        short_breakdown = (close[i] < s4_aligned[i]) and (vol_ratio[i] > 1.5) and (rsi[i] < 50)
        
        # Exit when price returns to opposite Camarilla level (mean reversion)
        exit_long = close[i] < s4_aligned[i] and position == 1
        exit_short = close[i] > r4_aligned[i] and position == -1
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakdown and position != -1:
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