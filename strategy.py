#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Trend_Volume_Confirm
Hypothesis: Daily Camarilla R3/S3 breakouts with 1-week EMA50 trend filter and volume spike (>2x average volume). 
In bull markets: price breaks above R3 with 1w uptrend and high volume → long. 
In bear markets: price breaks below S3 with 1w downtrend and high volume → short. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe.
Requires BTC/ETH edge via 1w trend and volume filters; avoids SOL-only bias by requiring trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 1 for Camarilla calculation, 50 for EMA)
    start_idx = max(1, 50)
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels using previous day's OHLC
        prev_1d_idx = i - 1
        if prev_1d_idx < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        prev_high = high[prev_1d_idx]
        prev_low = low[prev_1d_idx]
        prev_close = close[prev_1d_idx]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R3 and S3 levels
        R3 = prev_close + (range_val * 1.1 / 4)
        S3 = prev_close - (range_val * 1.1 / 4)
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1w_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(R3) or np.isnan(S3) or np.isnan(ema_val) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long logic: price breaks above R3 with 1w uptrend and volume confirmation
        long_condition = (close_val > R3) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S3 with 1w downtrend and volume confirmation
        short_condition = (close_val < S3) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal or opposite breakout
        exit_long = (close_val < ema_val) or (close_val < S3)
        exit_short = (close_val > ema_val) or (close_val > R3)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Camarilla_Pivot_Trend_Volume_Confirm"
timeframe = "1d"
leverage = 1.0