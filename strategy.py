#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Choppiness Index (CHOP) regime filter combined with 1-week ADX trend strength.
# CHOP > 61.8 = range-bound (mean reversion), CHOP < 38.2 = trending (trend following).
# In trending regime (CHOP < 38.2): use weekly ADX > 25 to confirm trend strength.
# Long: CHOP < 38.2 AND weekly ADX > 25 AND price > daily EMA50
# Short: CHOP < 38.2 AND weekly ADX > 25 AND price < daily EMA50
# Exit: CHOP > 50 (exit trending regime) or opposite signal
# This avoids whipsaw in ranging markets and captures strong trends.
# Target: 15-25 trades/year per symbol.
name = "1d_CHOP_ADX_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily EMA50 for trend bias
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Choppiness Index (CHOP)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_max_min = max_high - min_low
    chop = np.zeros(n)
    for i in range(n):
        if range_max_min[i] > 0:
            chop[i] = 100 * np.log10(atr[i] * np.sqrt(atr_period) / range_max_min[i]) / np.log10(atr_period)
        else:
            chop[i] = 50  # Neutral when no range
    
    # Weekly ADX for trend strength (using 1-week data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_1w))
    minus_dm = np.zeros(len(high_1w))
    tr_1w = np.zeros(len(high_1w))
    
    for i in range(1, len(high_1w)):
        high_diff = high_1w[i] - high_1w[i-1]
        low_diff = low_1w[i-1] - low_1w[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr_1w[i] = max(high_1w[i] - low_1w[i], 
                       abs(high_1w[i] - close_1w[i-1]), 
                       abs(low_1w[i] - close_1w[i-1]))
    
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr_1w
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr_1w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14*2)  # Ensure all indicators are valid
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50[i]) or np.isnan(chop[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop[i]
        adx_val = adx_aligned[i]
        price = close[i]
        ema = ema50[i]
        
        if position == 0:
            # Enter only in trending regime with strong trend
            if chop_val < 38.2 and adx_val > 25:
                if price > ema:
                    signals[i] = 0.25
                    position = 1
                elif price < ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: leave trending regime or reverse signal
            if chop_val > 50 or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: leave trending regime or reverse signal
            if chop_val > 50 or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals