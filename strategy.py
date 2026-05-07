#!/usr/bin/env python3
"""
1h_RSI_Momentum_4hTrend_Filter_v1
Hypothesis: Use RSI(14) momentum on 1h for entry timing, filtered by 4h EMA(50) trend direction.
Long when 1h RSI crosses above 50 and 4h EMA(50) is rising; short when RSI crosses below 50 and 4h EMA(50) is falling.
Add volume confirmation (volume > 1.5x 20-period average) and session filter (08-20 UTC) to reduce noise.
Target: 15-37 trades/year (60-150 total over 4 years) by using 4h trend for direction and 1h only for timing.
"""
name = "1h_RSI_Momentum_4hTrend_Filter_v1"
timeframe = "1h"
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
    
    # Pre-calculate session filter (08-20 UTC)
    hours = prices.index.hour  # already datetime64[ms], .hour works
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h EMA(50)
    close_4h = pd.Series(df_4h['close'])
    ema_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 4h EMA slope (rising/falling)
    ema_slope = np.zeros_like(ema_4h_aligned)
    ema_slope[1:] = ema_4h_aligned[1:] - ema_4h_aligned[:-1]
    # Rising when slope > 0, falling when slope < 0
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 14)  # Ensure sufficient warmup for RSI and volume
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades to reduce frequency (6h on 1h TF)
            if bars_since_exit < 6:
                continue
                
            # Long: RSI crosses above 50 and 4h EMA rising
            if (rsi[i] > 50 and rsi[i-1] <= 50 and 
                ema_slope[i] > 0 and volume_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_exit = 0
            # Short: RSI crosses below 50 and 4h EMA falling
            elif (rsi[i] < 50 and rsi[i-1] >= 50 and 
                  ema_slope[i] < 0 and volume_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: RSI returns to opposite side of 50
            if position == 1 and rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals