#!/usr/bin/env python3
# 4h_rsi_ema_crossover_1d_trend_volume_v2
# Hypothesis: RSI(3) + EMA(21) crossover on 4h with 1d EMA(100) trend filter and volume confirmation.
# Long when RSI(3) > 70 and EMA(21) > EMA(55) and price > 1d EMA(100) and volume > 1.5x average.
# Short when RSI(3) < 30 and EMA(21) < EMA(55) and price < 1d EMA(100) and volume > 1.5x average.
# Exit when RSI(3) crosses back to neutral (40 for long exit, 60 for short exit).
# Designed to capture momentum bursts with trend alignment in both bull and bear markets.
# Target: 75-200 total trades over 4 years (~19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_ema_crossover_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA100 for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate EMA21 and EMA55 for trend
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Calculate RSI(3)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    avg_loss = loss.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 55
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_21[i]) or np.isnan(ema_55[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI(3) crosses below 40
            if rsi_values[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI(3) crosses above 60
            if rsi_values[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Momentum entries: RSI extreme + EMA crossover + trend filter
            if (rsi_values[i] > 70) and (ema_21[i] > ema_55[i]) and (close[i] > ema_100_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (rsi_values[i] < 30) and (ema_21[i] < ema_55[i]) and (close[i] < ema_100_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals