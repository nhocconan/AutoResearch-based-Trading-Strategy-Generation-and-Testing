#!/usr/bin/env python3
# 4h_1d_RSI_Stoch_Confluence_LongOnly
# Hypothesis: Long-only strategy using RSI(14) oversold and Stochastic(14,3,3) oversold confluence on 4h,
# filtered by 1D EMA50 trend (price > EMA50 = bullish regime). Exits when RSI > 60 or price < EMA50.
# Works in bull via trend filter; in bear, avoids longs during downtrends, reducing whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Stoch_Confluence_LongOnly"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate EMA50 on 1d close for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h: RSI(14) ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h: Stochastic(14,3,3) ===
    high = prices['high'].values
    low = prices['low'].values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / np.where(highest_high - lowest_low > 0, highest_high - lowest_low, 1e-10)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    d_percent_smooth = pd.Series(d_percent).rolling(window=3, min_periods=3).mean().values  # Slow %D
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = close[i]
        rsi_val = rsi[i]
        stoch_val = d_percent_smooth[i]
        ema_50_val = ema_50_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(stoch_val) or np.isnan(ema_50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 AND Stoch < 30 AND price > EMA50 (bullish regime)
            if (rsi_val < 30 and 
                stoch_val < 30 and 
                close_val > ema_50_val):
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Long exit: RSI > 60 OR price < EMA50 (trend change or overbought)
            if (rsi_val > 60 or 
                close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals