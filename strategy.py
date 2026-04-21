#!/usr/bin/env python3
"""
6h_1w_1d_RSI_Divergence_Trend_Filter
Hypothesis: Weekly RSI divergence with daily trend filter for mean-reversion entries on 6h.
Long when: weekly RSI makes higher low while price makes lower low (bullish divergence)
           AND daily close > daily SMA50 (uptrend filter)
           AND price touches 6h Bollinger Lower Band (2,20) with rejection.
Short when: weekly RSI makes lower high while price makes higher high (bearish divergence)
            AND daily close < daily SMA50 (downtrend filter)
            AND price touches 6h Bollinger Upper Band (2,20) with rejection.
Exit on opposite band touch or RSI divergence failure.
Works in bull/bear: uses weekly divergence for exhaustion + daily trend filter to avoid counter-trend.
Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for RSI divergence
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.values
    
    # Align weekly RSI to 6h
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily SMA50
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # 6h Bollinger Bands (20, 2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    upper = upper.values
    lower = lower.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(sma_50_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # Detect weekly RSI divergence (need prior week)
        if i >= len(df_1w) * 28:  # Need at least 2 weeks of 6h data (approx)
            # Get current and prior weekly values (simplified: use current and 4 periods back ~1 week)
            if i >= 4:
                rsi_now = rsi_1w_aligned[i]
                rsi_prev = rsi_1w_aligned[i-4]
                price_now = price
                price_prev = close[i-4]
                
                # Bullish divergence: price lower low, RSI higher low
                bull_div = (price_now < price_prev) and (rsi_now > rsi_prev)
                # Bearish divergence: price higher high, RSI lower high
                bear_div = (price_now > price_prev) and (rsi_now < rsi_prev)
            else:
                bull_div = False
                bear_div = False
        else:
            bull_div = False
            bear_div = False
        
        if position == 0:
            # Long: bullish divergence + uptrend + touch lower band
            if bull_div and (close_1d[-1] > sma_50[-1] if len(close_1d) > 0 else False) and price <= lower[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + downtrend + touch upper band
            elif bear_div and (close_1d[-1] < sma_50[-1] if len(close_1d) > 0 else False) and price >= upper[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: touch upper band or divergence fails
            if price >= upper[i] or not bull_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: touch lower band or divergence fails
            if price <= lower[i] or not bear_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_1d_RSI_Divergence_Trend_Filter"
timeframe = "6h"
leverage = 1.0