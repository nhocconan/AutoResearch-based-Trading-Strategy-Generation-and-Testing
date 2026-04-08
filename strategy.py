#!/usr/bin/env python3
# 6h_ema_rsi_volume_trend
# Hypothesis: EMA trend filter + RSI momentum + volume confirmation on 6h timeframe.
# Long when price > EMA20, RSI > 50 and rising, and volume > 1.5x average.
# Short when price < EMA20, RSI < 50 and falling, and volume > 1.5x average.
# Uses 12h EMA trend filter for higher timeframe confirmation.
# Designed to capture momentum in both bull and bear markets with proper risk control.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_rsi_volume_trend"
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
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA20 on 6h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate RSI slope (momentum)
    rsi_slope = pd.Series(rsi_values).diff().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_20[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(rsi_slope[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below EMA20 OR RSI < 40
            if (close[i] < ema_20[i]) or (rsi_values[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above EMA20 OR RSI > 60
            if (close[i] > ema_20[i]) or (rsi_values[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Entry conditions with 12h trend filter
            if (close[i] > ema_20[i]) and (rsi_values[i] > 50) and (rsi_slope[i] > 0) and \
               (close[i] > ema_50_12h_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (close[i] < ema_20[i]) and (rsi_values[i] < 50) and (rsi_slope[i] < 0) and \
                 (close[i] < ema_50_12h_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals