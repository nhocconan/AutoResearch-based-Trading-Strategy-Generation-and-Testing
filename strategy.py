#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d EMA34 trend filter and volume-weighted RSI confirmation.
# Uses 1d EMA34 for trend direction and 12h RSI(14) with volume weighting for momentum.
# Enters only during 08-20 UTC session to avoid low-volume noise.
# Targets 20-30 trades/year (80-120 total over 4 years) with strict entry conditions.
# Works in bull/bear by following higher timeframe trends.
name = "12h_1d_EMA34_VolumeWeightedRSI"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume-weighted RSI calculation
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Apply volume weighting to gains and losses
    weighted_gain = gain * volume
    weighted_loss = loss * volume
    
    # Calculate weighted average gain and loss over 14 periods
    avg_weighted_gain = pd.Series(weighted_gain).rolling(window=14, min_periods=14).mean().values
    avg_weighted_loss = pd.Series(weighted_loss).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI
    rs = avg_weighted_gain / (avg_weighted_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for RSI calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d EMA34 AND RSI crosses above 30 (from oversold)
            if (close[i] > ema_34_1d_aligned[i] and 
                rsi[i] > 30 and rsi[i-1] <= 30):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34 AND RSI crosses below 70 (from overbought)
            elif (close[i] < ema_34_1d_aligned[i] and 
                  rsi[i] < 70 and rsi[i-1] >= 70):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d EMA34 or RSI crosses below 50
            if close[i] < ema_34_1d_aligned[i] or (rsi[i] < 50 and rsi[i-1] >= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d EMA34 or RSI crosses above 50
            if close[i] > ema_34_1d_aligned[i] or (rsi[i] > 50 and rsi[i-1] <= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals