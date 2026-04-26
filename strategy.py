#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Trend_Volume_Confirm
Hypothesis: On 12h timeframe, trade Camarilla R4/S4 breakouts with 1d EMA50 trend filter and volume spike (>1.5x average volume). 
In bull markets: price breaks above R4 with 1d uptrend and high volume → long. 
In bear markets: price breaks below S4 with 1d downtrend and high volume → short. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
Requires BTC/ETH edge via 1d trend and volume filters; avoids SOL-only bias by requiring trend alignment.
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
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels using previous day's OHLC
        # For 12h timeframe, 1 day = 2 bars
        prev_day_idx = i - 2
        if prev_day_idx < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        prev_high = high[prev_day_idx]
        prev_low = low[prev_day_idx]
        prev_close = close[prev_day_idx]
        
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
            
        # Camarilla R4 and S4 levels
        R4 = prev_close + (range_val * 1.1 / 2)
        S4 = prev_close - (range_val * 1.1 / 2)
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(R4) or np.isnan(S4) or np.isnan(ema_val) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume (balanced for 12h)
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above R4 with 1d uptrend and volume confirmation
        long_condition = (close_val > R4) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S4 with 1d downtrend and volume confirmation
        short_condition = (close_val < S4) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal or opposite breakout
        exit_long = (close_val < ema_val) or (close_val < S4)
        exit_short = (close_val > ema_val) or (close_val > R4)
        
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

name = "12h_Camarilla_Pivot_Trend_Volume_Confirm"
timeframe = "12h"
leverage = 1.0