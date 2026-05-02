#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter, volume spike (>1.5x average), and session filter (08-20 UTC)
# Uses 4h HTF for trend alignment and daily Camarilla levels. Volume threshold at 1.5x to balance trade frequency.
# Discrete position sizing 0.20 to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Session filter reduces noise trades outside active market hours.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Volume_Session"
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
    
    # Calculate Camarilla levels for R1 and S1 (using prior day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Prior day high
    prev_low = df_1d['low'].shift(1).values    # Prior day low
    prev_close = df_1d['close'].shift(1).values # Prior day close
    
    # Camarilla R1 and S1 levels (closer to mean than R3/S3)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (they update daily)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 1.5x 20-period average (balanced to avoid too few/many trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R1 AND price > 4h EMA50 AND volume spike AND in session
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 AND price < 4h EMA50 AND volume spike AND in session
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below S1 OR price < 4h EMA50
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price rises above R1 OR price > 4h EMA50
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals