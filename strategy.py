#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and volatility filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14) for volatility regime filter
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), np.abs(high_1w[1:] - close_1w[:-1]))
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly EMA(50) for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Donchian(20) for breakout signals
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily timeframe
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly ATR percentile (20-period) for volatility regime
    atr_percentile = pd.Series(atr14_1w_aligned).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Volatility regime: low volatility (ATR percentile < 30%) for breakout
    low_vol_regime = atr_percentile < 0.3
    
    # Trend filter: price above/below weekly EMA50
    uptrend = close > ema50_1w_aligned
    downtrend = close < ema50_1w_aligned
    
    # Breakout signals
    long_breakout = close > high_20
    short_breakout = close < low_20
    
    # Entry conditions: breakout in direction of trend during low volatility
    long_entry = long_breakout & uptrend & low_vol_regime
    short_entry = short_breakout & downtrend & low_vol_regime
    
    # Exit conditions: opposite Donchian breakout or trend reversal
    long_exit = (close < low_20) | (~uptrend)
    short_exit = (close > high_20) | (~downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr14_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(atr_percentile[i])):
            signals[i] = 0.0
            continue
        
        if long_entry[i] and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry[i] and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit[i] and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit[i] and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyATR_EMA50_Donchian20_Breakout_v1"
timeframe = "1d"
leverage = 1.0