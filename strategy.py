#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_1d_volatility_filter_v1
Hypothesis: On 4h timeframe, use daily Camarilla pivot levels to identify high-probability mean-reversion opportunities. Enter long when price touches S3 with rejection (close > open) and daily volatility is low (ATR percentile < 40). Enter short when price touches R3 with rejection (close < open) and daily volatility is low. Exit when price reaches opposite pivot level (S1/R1) or volatility increases (ATR percentile > 60). Uses volatility filter to avoid whipsaws in high volatility regimes. Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag while capturing mean reversion in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Pivot_1d_volatility_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation: based on previous day's range
    # R4 = close + (high - low) * 1.500
    # R3 = close + (high - low) * 1.250
    # R2 = close + (high - low) * 1.166
    # R1 = close + (high - low) * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.083
    # S2 = close - (high - low) * 1.166
    # S3 = close - (high - low) * 1.250
    # S4 = close - (high - low) * 1.500
    
    # Calculate for each day
    rang = high_1d - low_1d
    c = close_1d
    
    r3 = c + rang * 1.250
    r1 = c + rang * 1.083
    s1 = c - rang * 1.083
    s3 = c - rang * 1.250
    
    # Align to 4h timeframe (shifted by 1 day to avoid look-ahead)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d ATR for volatility regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile rank (50-day lookback for reasonable sampling)
    atr_percentile = pd.Series(atr_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    atr_percentile_4h = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(r3_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(s3_4h[i]) or np.isnan(atr_percentile_4h[i]) or 
            np.isnan(close[i]) or np.isnan(open_price[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR percentile below 40th percentile
        low_vol = atr_percentile_4h[i] < 0.4
        
        # Price rejection conditions: close > open for bullish rejection, close < open for bearish rejection
        bullish_rejection = close[i] > open_price[i]
        bearish_rejection = close[i] < open_price[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 or volatility increases
            if close[i] <= s1_4h[i] or atr_percentile_4h[i] > 0.6:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 or volatility increases
            if close[i] >= r1_4h[i] or atr_percentile_4h[i] > 0.6:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if low_vol:
                # Long entry: price touches S3 with bullish rejection
                if low_vol and bullish_rejection and close[i] <= s3_4h[i] * 1.001:  # Allow small buffer
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches R3 with bearish rejection
                elif low_vol and bearish_rejection and close[i] >= r3_4h[i] * 0.999:  # Allow small buffer
                    position = -1
                    signals[i] = -0.25
    
    return signals