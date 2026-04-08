#!/usr/bin/env python3
# 1h_4d_1d_rsi_reversal_v1
# Hypothesis: Trade reversals at RSI extremes with trend filter from 4h EMA50 and 1d trend filter.
# In 1d uptrend: go long when RSI(14) < 30 on 1h, exit when RSI > 70 or 1d trend breaks.
# In 1d downtrend: go short when RSI(14) > 70 on 1h, exit when RSI < 30 or 1d trend breaks.
# Uses volume confirmation to avoid false signals. Target: 15-35 trades/year (60-140 total over 4 years).
# Uses 4h EMA50 for intermediate trend filter and 1d EMA50 for higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_rsi_reversal_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for intermediate trend filter
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA50 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or 1d trend breaks (price < 1d EMA50)
            if rsi[i] > 70 or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or 1d trend breaks (price > 1d EMA50)
            if rsi[i] < 30 or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: RSI < 30 with volume surge and 4d/1d uptrend
            if (rsi[i] < 30 and vol_surge and 
                close[i] > ema50_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 70 with volume surge and 4d/1d downtrend
            elif (rsi[i] > 70 and vol_surge and 
                  close[i] < ema50_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals