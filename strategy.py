#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolRegime_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h with 4h EMA50 trend filter and 1d volume regime filter.
Only trade long when 4h EMA50 up + 1d volume above average, short when 4h EMA50 down + 1d volume above average.
Use 1h Camarilla levels from prior 4h bar for structure. Volume filter reduces false breakouts in low volatility.
Target: 20-50 trades/year to minimize fee drag while capturing high-probability breakouts.
Discrete sizing: 0.20.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate prior 4h bar OHLC for Camarilla levels
    # Camarilla uses prior day's OHLC, but we use prior 4h bar for 1h timeframe
    prior_4h_high = df_4h['high'].shift(1).values  # prior completed 4h bar
    prior_4h_low = df_4h['low'].shift(1).values
    prior_4h_close = df_4h['close'].shift(1).values
    
    # Calculate Camarilla levels for prior 4h bar
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_range = prior_4h_high - prior_4h_low
    r1 = prior_4h_close + camarilla_range * 1.1 / 12
    s1 = prior_4h_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (delayed by one 4h bar for completion)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period SMA)
    avg_vol_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 4h EMA50 (50), Camarilla (need prior 4h bar), 1d avg volume (20)
    start_idx = max(50, 20) + 1  # +1 for prior 4h bar shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume regime: 1d volume above average (avoid low volatility breakouts)
        volume_regime = volume[i] > avg_vol_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above R1, 4h EMA50 up, volume regime
            if (close[i] > r1_aligned[i] and 
                close[i-1] <= r1_aligned[i-1] and  # ensure breakout happened this bar
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and  # 4h EMA50 rising
                volume_regime):
                signals[i] = 0.20
                position = 1
            
            # Short setup: price breaks below S1, 4h EMA50 down, volume regime
            elif (close[i] < s1_aligned[i] and 
                  close[i-1] >= s1_aligned[i-1] and  # ensure breakdown happened this bar
                  ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and  # 4h EMA50 falling
                  volume_regime):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price breaks below S1 OR 4h EMA50 turns down OR end of session
            if (close[i] < s1_aligned[i] or 
                ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] or
                hour >= 20):  # force exit before session end
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price breaks above R1 OR 4h EMA50 turns up OR end of session
            if (close[i] > r1_aligned[i] or 
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] or
                hour >= 20):  # force exit before session end
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolRegime_v1"
timeframe = "1h"
leverage = 1.0