#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_Strength_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h and 1d data once
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d RSI for overbought/oversold
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # 1h Bollinger Bands for entry timing
    close_1h = prices['close'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    sma_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_vals = upper_bb.values
    lower_bb_vals = lower_bb.values
    
    # Pre-compute hour filter
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(upper_bb_vals[i]) or np.isnan(lower_bb_vals[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend + price near lower BB + RSI not overbought
            if (close_1h[i] > ema_50_4h_aligned[i] and 
                close_1h[i] <= lower_bb_vals[i] * 1.01 and  # Allow small tolerance
                rsi_1d_aligned[i] < 70):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + price near upper BB + RSI not oversold
            elif (close_1h[i] < ema_50_4h_aligned[i] and 
                  close_1h[i] >= upper_bb_vals[i] * 0.99 and  # Allow small tolerance
                  rsi_1d_aligned[i] > 30):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Exit long: 4h downtrend or price touches upper BB
            if (close_1h[i] < ema_50_4h_aligned[i] or 
                close_1h[i] >= upper_bb_vals[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Exit short: 4h uptrend or price touches lower BB
            if (close_1h[i] > ema_50_4h_aligned[i] or 
                close_1h[i] <= lower_bb_vals[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals