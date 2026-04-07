#!/usr/bin/env python3
"""
6h_adaptive_keltner_1w_trend_v1
Hypothesis: On 6-hour timeframe, use weekly ATR-based Keltner Channels with adaptive bands (wider in high volatility) and trend filter from 1-day EMA200. Enter long when price touches lower band in uptrend, short when price touches upper band in downtrend. Exit when price crosses middle band (EMA10) or reverses. Designed for low frequency (15-25 trades/year) to avoid fee drag while capturing mean reversion in trends. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) by adapting to trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adaptive_keltner_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_keltner(high, low, close, atr_multiplier=2.0):
    """Calculate Keltner Channel: middle=EMA(20), upper/lower=middle ± ATR*mult"""
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # EMA20 as middle band
    ema20 = close_series.ewm(span=20, adjust=False).mean()
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(10)
    atr = tr.ewm(span=10, adjust=False).mean()
    
    # Bands
    upper = ema20 + (atr * atr_multiplier)
    lower = ema20 - (atr * atr_multiplier)
    
    return ema20.values, upper.values, lower.values, atr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for Keltner Channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Calculate weekly Keltner with adaptive multiplier based on volatility regime
    w_close_series = pd.Series(w_close)
    w_ema200 = w_close_series.ewm(span=200, adjust=False).mean()
    w_price_above_ema200 = w_close > w_ema200.values
    
    # Adaptive multiplier: 1.5 in low vol (trending), 2.5 in high vol (choppy)
    # Use weekly ATR(20) to determine regime
    w_tr1 = w_high - w_low
    w_tr2 = abs(w_high - w_close_series.shift(1))
    w_tr3 = abs(w_low - w_close_series.shift(1))
    w_tr = pd.concat([w_tr1, w_tr2, w_tr3], axis=1).max(axis=1)
    w_atr = w_tr.ewm(span=20, adjust=False).mean()
    w_atr_ma = w_atr.ewm(span=50, adjust=False).mean()
    w_volatility_ratio = w_atr / w_atr_ma
    w_volatility_ratio = w_volatility_ratio.fillna(1.0)
    
    # Adaptive multiplier: lower when volatility is low (trending), higher when high (choppy)
    w_adaptive_mult = 1.5 + (w_volatility_ratio - 1.0)  # ranges 1.5 to 2.5
    w_adaptive_mult = np.clip(w_adaptive_mult, 1.5, 2.5)
    
    # Calculate Keltner with adaptive multiplier
    keltner_data = np.array([calculate_keltner(w_high[i], w_low[i], w_close[i], w_adaptive_mult[i]) 
                             for i in range(len(w_close))])
    # Columns: middle, upper, lower, atr
    w_keltner_middle = keltner_data[:, 0]
    w_keltner_upper = keltner_data[:, 1]
    w_keltner_lower = keltner_data[:, 2]
    
    # Align to 6h timeframe
    w_keltner_middle_aligned = align_htf_to_ltf(prices, df_1w, w_keltner_middle)
    w_keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, w_keltner_upper)
    w_keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, w_keltner_lower)
    
    # Daily EMA200 for trend filter (more responsive than weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    d_ema200 = pd.Series(d_close).ewm(span=200, adjust=False).mean().values
    d_ema200_aligned = align_htf_to_ltf(prices, df_1d, d_ema200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if weekly Keltner not available
        if np.isnan(w_keltner_middle_aligned[i]) or np.isnan(w_keltner_upper_aligned[i]) or np.isnan(w_keltner_lower_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if daily EMA200 not available
        if np.isnan(d_ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA200
        uptrend = close[i] > d_ema200_aligned[i]
        downtrend = close[i] < d_ema200_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses above middle band (take profit)
            if close[i] > w_keltner_middle_aligned[i]:
                exit_long = True
            # Exit when trend turns down
            elif not uptrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses below middle band (take profit)
            if close[i] < w_keltner_middle_aligned[i]:
                exit_short = True
            # Exit when trend turns up
            elif not downtrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches or goes below lower band in uptrend
            long_entry = (close[i] <= w_keltner_lower_aligned[i]) and uptrend
            # Short entry: price touches or goes above upper band in downtrend
            short_entry = (close[i] >= w_keltner_upper_aligned[i]) and downtrend
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals