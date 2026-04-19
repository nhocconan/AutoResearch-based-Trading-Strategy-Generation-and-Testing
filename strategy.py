#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA direction with 1d RSI filter and 1w ADX trend filter
# - KAMA(10) on 12h: long when price > KAMA, short when price < KAMA
# - 1d RSI(14) for momentum confirmation: long when RSI > 50, short when RSI < 50
# - 1w ADX(14) for trend strength: only trade when ADX > 25 (strong trend)
# - Exit when KAMA direction reverses or ADX falls below 20
# - Designed to capture trends in both bull and bear markets using adaptive smoothing
# - Target: 15-25 trades/year to minimize fee drag

name = "12h_KAMA_1dRSI_1wADX_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1d
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1w data for ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1w
    plus_dm = pd.Series(high_1w).diff()
    minus_dm = pd.Series(low_1w).diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0.0)
    minus_dm = minus_dm.where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0.0)
    
    tr1 = pd.Series(high_1w).subtract(pd.Series(low_1w))
    tr2 = pd.Series(high_1w).subtract(pd.Series(close_1w).shift(1)).abs()
    tr3 = pd.Series(low_1w).subtract(pd.Series(close_1w).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_1w = tr.rolling(window=14, min_periods=14).mean()
    plus_di_1w = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr_1w)
    minus_di_1w = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr_1w)
    dx_1w = (abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)) * 100
    adx_1w = dx_1w.rolling(window=14, min_periods=14).mean()
    adx_1w = adx_1w.fillna(0).values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate KAMA(10) on 12h
    # Efficiency Ratio
    change = abs(pd.Series(close).diff(10))
    volatility = pd.Series(close).diff().abs().rolling(window=10).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = [np.nan] * len(close)
    kama[9] = close[9]  # Seed
    
    for i in range(10, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama = np.array(kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        trend_filter = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Look for long entry: price > KAMA + RSI > 50 + strong trend
            if close[i] > kama[i] and rsi_1d_aligned[i] > 50 and trend_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price < KAMA + RSI < 50 + strong trend
            elif close[i] < kama[i] and rsi_1d_aligned[i] < 50 and trend_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price < KAMA or RSI < 40 or ADX < 20
            if (close[i] < kama[i] or rsi_1d_aligned[i] < 40 or 
                adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price > KAMA or RSI > 60 or ADX < 20
            if (close[i] > kama[i] or rsi_1d_aligned[i] > 60 or 
                adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals