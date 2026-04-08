#!/usr/bin/env python3
# 12h_1d_1w_rsi_trend_v1
# Hypothesis: Trade with the weekly trend using RSI(14) on 12h for entry.
# In weekly uptrend: go long when RSI < 30 (oversold) with volume confirmation.
# In weekly downtrend: go short when RSI > 70 (overbought) with volume confirmation.
# Exit when RSI crosses 50 or weekly trend reverses.
# Uses weekly EMA21 for trend, daily volume for confirmation, and 12h RSI for timing.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_rsi_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 12h RSI(14)
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Daily volume surge condition (volume > 1.5x 20-day average)
        vol_surge = volume_1d[i // 2] > 1.5 * vol_ma_20_1d_aligned[i] if vol_ma_20_1d_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 or weekly trend breaks (price < weekly EMA21)
            if rsi[i] > 50 or close[i] < ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 or weekly trend breaks (price > weekly EMA21)
            if rsi[i] < 50 or close[i] > ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold) with volume surge and weekly uptrend
            if (rsi[i] < 30 and vol_surge and 
                close[i] > ema21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 (overbought) with volume surge and weekly downtrend
            elif (rsi[i] > 70 and vol_surge and 
                  close[i] < ema21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals