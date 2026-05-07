#!/usr/bin/env python3
# 1h_Volume_Regime_Trend
# Hypothesis: 1h strategy using 4h EMA trend filter, 1d volume regime filter, and 1h RSI mean reversion.
# Long when 4h EMA20 uptrend, 1d volume above average (liquidity), and 1h RSI < 40 (oversold).
# Short when 4h EMA20 downtrend, 1d volume above average, and 1h RSI > 60 (overbought).
# Uses 08-20 UTC session filter to avoid low-liquidity periods. Target: 15-30 trades/year per symbol.

name = "1h_Volume_Regime_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate EMA20 on 4h closes for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 20-day average volume on 1d
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 1h RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(100).values  # Fill NaN with 100 (no loss case)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 14)  # Ensure we have EMA, volume, and RSI data
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any critical value is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or vol_avg_1d_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h EMA uptrend, high volume regime, RSI oversold
            if (close[i] > ema_20_4h_aligned[i] and 
                volume[i] > vol_avg_1d_aligned[i] and 
                rsi_values[i] < 40):
                signals[i] = 0.20
                position = 1
            # Short: 4h EMA downtrend, high volume regime, RSI overbought
            elif (close[i] < ema_20_4h_aligned[i] and 
                  volume[i] > vol_avg_1d_aligned[i] and 
                  rsi_values[i] > 60):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: trend failure or RSI overbought
            if (close[i] < ema_20_4h_aligned[i] or 
                rsi_values[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: trend failure or RSI oversold
            if (close[i] > ema_20_4h_aligned[i] or 
                rsi_values[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals