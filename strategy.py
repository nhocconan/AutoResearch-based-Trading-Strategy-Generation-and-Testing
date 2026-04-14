#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points (from weekly high/low/close) as support/resistance.
# Long when price retraces to weekly pivot support (S1) with bullish divergence on RSI and volume confirmation.
# Short when price retraces to weekly pivot resistance (R1) with bearish divergence on RSI and volume confirmation.
# Exit when price reaches opposite pivot level or RSI shows exhaustion.
# Uses weekly pivots for structural levels, RSI divergence for momentum exhaustion, and volume for confirmation.
# Designed to work in both bull and bear markets by fading extreme moves at key weekly levels.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Support and resistance levels
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate RSI (14) for divergence detection
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 1.3x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need RSI and volume MA periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # RSI conditions for divergence (simplified)
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        if position == 0:
            # Look for retracement to weekly S1 with bullish signs
            # Long: price near S1, RSI oversold, volume confirmation
            if (low[i] <= s1_aligned[i] * 1.005 and  # Allow small buffer
                rsi_oversold and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Look for retracement to weekly R1 with bearish signs
            # Short: price near R1, RSI overbought, volume confirmation
            elif (high[i] >= r1_aligned[i] * 0.995 and   # Allow small buffer
                  rsi_overbought and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches pivot or RSI shows exhaustion
            if (close[i] >= pivot_aligned[i] * 0.995 or  # Near pivot
                rsi[i] > 65):  # RSI showing weakness
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches pivot or RSI shows exhaustion
            if (close[i] <= pivot_aligned[i] * 1.005 or   # Near pivot
                rsi[i] < 35):  # RSI showing weakness
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_Points_RSI_Divergence_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0