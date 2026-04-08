#!/usr/bin/env python3

# 1d_1w_volatility_breakout
# Hypothesis: Daily volatility breakout with weekly trend filter, volume confirmation, and ATR-based exit.
# Designed to work in both bull and bear markets by capturing volatility expansion phases.
# Target: 10-20 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_volatility_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (1w EMA50) - load once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on weekly data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily indicators
    # EMA20 for dynamic center line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for volatility bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Upper and lower bands (EMA20 ± 2*ATR)
    upper_band = ema20 + 2.0 * atr
    lower_band = ema20 - 2.0 * atr
    
    # Volume confirmation (20-day average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(avg_volume[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: close below EMA20 or volatility contraction
            if close[i] < ema20[i] or atr[i] < atr[i-1] * 0.8:  # Volatility dropping
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: close above EMA20 or volatility contraction
            if close[i] > ema20[i] or atr[i] < atr[i-1] * 0.8:  # Volatility dropping
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # Breakout conditions
            if volume_ok:
                # Long breakout: price crosses above upper band in uptrend
                if weekly_uptrend and close[i] > upper_band[i] and close[i-1] <= upper_band[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price crosses below lower band in downtrend
                elif weekly_downtrend and close[i] < lower_band[i] and close[i-1] >= lower_band[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals