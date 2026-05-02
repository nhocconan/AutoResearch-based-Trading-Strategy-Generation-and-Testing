#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter, volume spike (>1.8x average), and session filter (08-20 UTC)
# Uses 4h HTF for Camarilla levels and EMA50 to capture institutional structure and reduce false breakouts.
# Volume confirmation at 1.8x average ensures strong participation. Session filter avoids low-liquidity hours.
# Discrete sizing 0.20 to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Primary timeframe: 1h, HTF: 4h for Camarilla levels and EMA50.

name = "1h_Camarilla_H3_L3_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Calculate Camarilla levels H3 and L3 from 4h timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Prior 4h bar's high, low, close for Camarilla calculation
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Camarilla H3 and L3 levels
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 1.8x 20-period average (balanced threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = open_time.dt.hour.values
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above H3 AND price > 4h EMA50 AND volume spike AND session
            if (close[i] > camarilla_h3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below L3 AND price < 4h EMA50 AND volume spike AND session
            elif (close[i] < camarilla_l3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below L3 OR price < 4h EMA50
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price rises above H3 OR price > 4h EMA50
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals