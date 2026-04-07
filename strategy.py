#!/usr/bin/env python3
"""
6h_1d_volatility_breakout_v1
Hypothesis: On 6h timeframe, breakouts from volatility contractions (Bollinger Bands width < 20th percentile) are more likely to succeed when aligned with 1d trend (price > EMA50). Enter long on breakout above upper BB with volume confirmation, short on breakdown below lower BB. Exit on opposite band touch. Designed for 15-30 trades/year to minimize fee drag while capturing explosive moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2.0)
    if len(close) < 20:
        return np.zeros(n)
    
    # Basis (SMA)
    basis = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Deviation
    dev = pd.Series(close).rolling(window=20, min_periods=20).std(ddof=0).values
    
    # Upper and Lower Bands
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    
    # Bollinger Band Width
    bb_width = (upper - lower) / basis
    
    # Calculate 20-period percentile rank of BB width (volatility regime)
    # Using rolling percentile approximation: rank of current value in window
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(
        window=50, min_periods=50
    ).apply(
        lambda x: (pd.Series(x).rank(method='average').iloc[-1] - 1) / (len(x) - 1) * 100,
        raw=False
    ).values
    
    # Volatility contraction: BB width below 20th percentile
    vol_contract = bb_width_percentile < 20.0
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches lower band (mean reversion)
            if close[i] <= lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper band (mean reversion)
            if close[i] >= upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_contract[i] and vol_ok:
                # Long: breakout above upper band with 1d uptrend
                if close[i] > upper[i] and close[i-1] <= upper[i-1] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: breakdown below lower band with 1d downtrend
                elif close[i] < lower[i] and close[i-1] >= lower[i-1] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals