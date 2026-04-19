#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trading with 4h RSI momentum and 1d volume confirmation
# - Use 4h RSI(14) for momentum: long when >55, short when <45
# - Require 1d volume > 1.5x 20-period average for conviction
# - Only trade during active session (08-20 UTC) to avoid low volatility periods
# - Fixed position size of 0.20 to manage risk
# - Designed for 15-30 trades/year to minimize fee drag while capturing trends

name = "1h_RSI4h_Volume1d_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI momentum
    df_4h = get_htf_data(prices, '4h')
    
    # 4h RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume filter: current 1h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 1h: 1d has 24x 1h bars, so divide by 24
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 24.0)
        
        if position == 0 and in_session and volume_filter:
            # Look for long entry: bullish momentum (RSI > 55)
            if rsi_4h_aligned[i] > 55:
                signals[i] = 0.20
                position = 1
            # Look for short entry: bearish momentum (RSI < 45)
            elif rsi_4h_aligned[i] < 45:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish momentum or outside session
            if rsi_4h_aligned[i] < 45 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit on bullish momentum or outside session
            if rsi_4h_aligned[i] > 55 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals