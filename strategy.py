#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion + 4h trend filter + volume confirmation
# Uses 4h EMA50 for trend direction, 1h RSI for mean reversion entries, and volume spike to confirm.
# In uptrend (price > EMA50): buy when RSI < 30 (oversold), sell when RSI > 70 (overbought).
# In downtrend (price < EMA50): sell short when RSI > 70, cover when RSI < 30.
# Session filter (08-20 UTC) reduces noise. Target: 60-150 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period on 1h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg[i])):
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            continue
        
        # Long conditions: uptrend + RSI oversold + volume spike
        if (close[i] > ema50_4h_aligned[i] and  # Uptrend
            rsi[i] < 30 and                    # Oversold
            volume[i] > 1.5 * vol_avg[i] and   # Volume spike
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short conditions: downtrend + RSI overbought + volume spike
        elif (close[i] < ema50_4h_aligned[i] and  # Downtrend
              rsi[i] > 70 and                     # Overbought
              volume[i] > 1.5 * vol_avg[i] and    # Volume spike
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI mean reversion (return to neutral)
        elif position == 1 and rsi[i] > 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi[i] < 50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0