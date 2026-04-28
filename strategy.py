#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1_S1_Breakout_Trend_Volume
Hypothesis: Use 4h/1d trend and volume confirmation with daily Camarilla R1/S1 breakouts on 1h.
Targets 20-50 trades/year by requiring 4h trend alignment, 1d volume surge, and 1h breakout.
Works in bull/bear: long in 4h uptrend, short in 4h downtrend. Volume surge filters low-quality breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels (tighter breakout)
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d volume average for surge detection
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    # Align all higher timeframe data to 1h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Trend filters
    uptrend_4h = close > ema_34_4h_aligned
    downtrend_4h = close < ema_34_4h_aligned
    
    # Volume surge: current 1h volume > 2.0x 1d average volume (scaled to hourly)
    # Scale daily average to hourly approximation: divide by 16 (24h/1.5h per bar approx)
    vol_threshold = vol_avg_1d_aligned / 16.0 * 2.0  # 2x scaled hourly average
    volume_surge = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: price breaks above R1 + 4h uptrend + volume surge
        long_entry = (close[i] > R1_aligned[i] and 
                     uptrend_4h[i] and 
                     volume_surge[i])
        
        # Short: price breaks below S1 + 4h downtrend + volume surge
        short_entry = (close[i] < S1_aligned[i] and 
                      downtrend_4h[i] and 
                      volume_surge[i])
        
        # Exit on opposite level break
        long_exit = close[i] < S1_aligned[i]
        short_exit = close[i] > R1_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.20  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.20   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h1d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0