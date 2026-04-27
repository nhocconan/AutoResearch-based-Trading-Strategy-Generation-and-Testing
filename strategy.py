#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ATR(14) for volatility filter
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly pivot points (classic)
    pivot_1w = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3.0
    r1_1w = 2 * pivot_1w - low_1w[:-1]
    s1_1w = 2 * pivot_1w - high_1w[:-1]
    r2_1w = pivot_1w + (high_1w[:-1] - low_1w[:-1])
    s2_1w = pivot_1w - (high_1w[:-1] - low_1w[:-1])
    r3_1w = high_1w[:-1] + 2 * (pivot_1w - low_1w[:-1])
    s3_1w = low_1w[:-1] - 2 * (high_1w[:-1] - pivot_1w)
    
    # Align weekly pivot points to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Calculate 6h RSI(14) for momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[np.nan], gain])
    loss = np.concatenate([[np.nan], loss])
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_1w_aligned[i] > np.nanpercentile(atr_1w_aligned[:i+1], 30)
        
        # Mean reversion at extreme weekly pivot levels
        near_r3 = abs(close[i] - r3_1w_aligned[i]) / close[i] < 0.015  # Within 1.5% of R3
        near_s3 = abs(close[i] - s3_1w_aligned[i]) / close[i] < 0.015  # Within 1.5% of S3
        
        # RSI conditions for mean reversion
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        # Long: price near S3 + oversold RSI + volatility filter
        long_condition = near_s3 and rsi_oversold and vol_filter
        
        # Short: price near R3 + overbought RSI + volatility filter
        short_condition = near_r3 and rsi_overbought and vol_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: return to pivot or RSI normalization
        elif position == 1 and (close[i] >= pivot_1w_aligned[i] or rsi[i] > 50):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] <= pivot_1w_aligned[i] or rsi[i] < 50):
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

name = "6h_WeeklyPivot_RSI_MeanReversion"
timeframe = "6h"
leverage = 1.0