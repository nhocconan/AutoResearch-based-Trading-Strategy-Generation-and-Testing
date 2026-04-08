#!/usr/bin/env python3
# 6h_rsi_slope_reversal_1d_trend
# Hypothesis: On 6h, use RSI slope (3-bar change) to detect momentum exhaustion near extremes.
# Long when RSI slope turns positive from oversold (RSI<30) with 1d uptrend (price > EMA50).
# Short when RSI slope turns negative from overbought (RSI>70) with 1d downtrend (price < EMA50).
# Uses 1d EMA50 for trend filter. Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_slope_reversal_1d_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI (14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate RSI slope (3-bar change)
    rsi_series = pd.Series(rsi)
    rsi_slope = rsi_series.diff(3).values  # positive = rising momentum
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30  # need 14 for RSI + 3 for slope
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(rsi_slope[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI slope turns negative (momentum fading) OR RSI > 70
            if rsi_slope[i] < 0 or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI slope turns positive (momentum fading) OR RSI < 30
            if rsi_slope[i] > 0 or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: RSI slope turns positive from oversold with uptrend
            if (rsi_slope[i] > 0 and rsi[i] < 30 and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: RSI slope turns negative from overbought with downtrend
            elif (rsi_slope[i] < 0 and rsi[i] > 70 and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals