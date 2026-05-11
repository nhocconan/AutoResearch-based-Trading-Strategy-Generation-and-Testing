#!/usr/bin/env python3
name = "1d_1W_RSI_Bollinger_Reversion"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly close for trend filter
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Bollinger Bands(20, 2)
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = ma + 2 * std
    lower = ma - 2 * std
    
    # Volume ratio (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_surge = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price at/below lower BB + weekly uptrend + volume surge
            if (rsi[i] < 30 and 
                close[i] <= lower[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + price at/above upper BB + weekly downtrend + volume surge
            elif (rsi[i] > 70 and 
                  close[i] >= upper[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI reverts to neutral (40-60) or opposite extreme
            if position == 1:
                # Exit long: RSI > 50 or price crosses above middle band
                if (rsi[i] > 50) or (close[i] >= ma[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI < 50 or price crosses below middle band
                if (rsi[i] < 50) or (close[i] <= ma[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals