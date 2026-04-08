#!/usr/bin/env python3
# 1h_rsi_ema_crossover_4h_trend_volume
# Hypothesis: On 1h timeframe, enter long when RSI crosses above 30 and price > 4h EMA50 with volume > 1.5x average.
# Enter short when RSI crosses below 70 and price < 4h EMA50 with volume > 1.5x average.
# Exit when RSI crosses back to neutral (50) or volume condition fails.
# Uses 4h EMA for trend filter to avoid counter-trend trades, volume confirmation to avoid false signals.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 100-180 total trades over 4 years (~25-45/year) within 1h limits.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_ema_crossover_4h_trend_volume"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 or volume condition fails
            if rsi[i] < 50 or volume[i] <= 1.5 * avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 or volume condition fails
            if rsi[i] > 50 or volume[i] <= 1.5 * avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # RSI crossover entries with 4h trend filter
            if (rsi[i] > 30 and rsi[i-1] <= 30 and 
                close[i] > ema_50_4h_aligned[i] and volume_ok):
                position = 1
                signals[i] = 0.20
            elif (rsi[i] < 70 and rsi[i-1] >= 70 and 
                  close[i] < ema_50_4h_aligned[i] and volume_ok):
                position = -1
                signals[i] = -0.20
    
    return signals