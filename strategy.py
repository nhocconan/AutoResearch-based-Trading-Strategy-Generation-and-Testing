#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR(14) for volatility filtering
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = abs(pd.Series(close_1d).shift(1) - pd.Series(high_1d))
    tr3 = abs(pd.Series(close_1d).shift(1) - pd.Series(low_1d))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Get weekly data for higher timeframe context (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/both EMAs
        price_above_both = close[i] > ema_50_1d_aligned[i] and close[i] > ema_50_1w_aligned[i]
        price_below_both = close[i] < ema_50_1d_aligned[i] and close[i] < ema_50_1w_aligned[i]
        
        # Volatility filter: ATR above median (avoid choppy markets)
        if i >= 20:
            atr_recent = atr_14_1d_aligned[i-20:i]
            atr_median = np.median(atr_recent[~np.isnan(atr_recent)]) if len(atr_recent[~np.isnan(atr_recent)]) > 0 else atr_14_1d_aligned[i]
            volatility_filter = atr_14_1d_aligned[i] > atr_median * 0.8
        else:
            volatility_filter = True
        
        # Long conditions: bullish trend + volatility filter
        long_condition = (price_above_both and volatility_filter)
        
        # Short conditions: bearish trend + volatility filter
        short_condition = (price_below_both and volatility_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_both:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_both:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_EMA50_1D_1W_Trend_Filter"
timeframe = "12h"
leverage = 1.0