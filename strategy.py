#!/usr/bin/env python3
# 1h_volume_spike_4h1d_trend
# Hypothesis: Volume spike breakouts with 4h/1d trend filter. Volume spikes (>2x avg) indicate strong momentum.
# Long when price breaks above recent high with volume spike and 4h/1d uptrend.
# Short when price breaks below recent low with volume spike and 4h/1d downtrend.
# Exit when price crosses back to 4h EMA20.
# Uses volume confirmation to avoid false breakouts and reduce trade frequency.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_spike_4h1d_trend"
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
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate average volume for spike detection (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate recent high/low for breakout detection (10-period)
    recent_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    recent_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(recent_high[i]) or np.isnan(recent_low[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 4h EMA20
            if close[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 4h EMA20
            if close[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume spike: current volume > 2.0x average volume
            volume_spike = volume[i] > 2.0 * avg_volume[i]
            
            # Breakout entries with volume confirmation
            # Long: break above recent high with volume spike and uptrend (price > both EMAs)
            if (close[i] > recent_high[i]) and volume_spike and \
               (close[i] > ema_20_4h_aligned[i]) and (close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: break below recent low with volume spike and downtrend (price < both EMAs)
            elif (close[i] < recent_low[i]) and volume_spike and \
                 (close[i] < ema_20_4h_aligned[i]) and (close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals