#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Supertrend for trend direction (ATR=10, mult=3)
    # True Range
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3 = np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    upper_band = hl2 + 3 * atr_4h
    lower_band = hl2 - 3 * atr_4h
    
    # Initialize Supertrend
    supertrend = np.zeros(len(df_4h))
    direction = np.ones(len(df_4h))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_4h)):
        if df_4h['close'].iloc[i] > upper_band[i-1]:
            direction[i] = 1
        elif df_4h['close'].iloc[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    
    # 1h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(rsi_14[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h Supertrend uptrend (price above Supertrend)
        # 2. 1h RSI oversold (< 30) for entry timing
        if (close[i] > supertrend_4h_aligned[i] and
            rsi_14[i] < 30):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h Supertrend downtrend (price below Supertrend)
        # 2. 1h RSI overbought (> 70) for entry timing
        elif (close[i] < supertrend_4h_aligned[i] and
              rsi_14[i] > 70):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Supertrend4h_RSI14_v1"
timeframe = "1h"
leverage = 1.0