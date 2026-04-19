#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI + Stochastic dual-oscillator confluence with 1-day trend filter.
# Long when: RSI(14) crosses above 30 (oversold bounce), Stoch %K crosses above %D, price > EMA50(1d)
# Short when: RSI(14) crosses below 70 (overbought rejection), Stoch %K crosses below %D, price < EMA50(1d)
# Exit when RSI returns to opposite threshold (RSI > 70 for longs, RSI < 30 for shorts)
# RSI captures momentum extremes, Stochastic confirms turning point, EMA50 filters trend direction.
# Designed for 15-35 trades/year per symbol. Works in bull (buy oversold dips) and bear (sell overbought rallies).

name = "6h_RSI_Stoch_EMA50_Confluence"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate RSI(14) on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic(14,3,3) on 6h data
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k_percent = np.where((highest_high - lowest_low) == 0, 50, k_percent)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(k_percent[i]) or np.isnan(d_percent[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        k_val = k_percent[i]
        d_val = d_percent[i]
        ema50 = ema50_1d_aligned[i]
        
        if position == 0:
            # Long entry: RSI crosses above 30, Stoch K crosses above D, price > EMA50
            if (rsi_val > 30 and rsi[i-1] <= 30 and 
                k_val > d_val and k_percent[i-1] <= d_percent[i-1] and
                price > ema50):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI crosses below 70, Stoch K crosses below D, price < EMA50
            elif (rsi_val < 70 and rsi[i-1] >= 70 and 
                  k_val < d_val and k_percent[i-1] >= d_percent[i-1] and
                  price < ema50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns above 70 (overbought)
            if rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns below 30 (oversold)
            if rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals