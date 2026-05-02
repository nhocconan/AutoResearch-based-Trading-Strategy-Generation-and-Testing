#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with 1d EMA50 trend filter and volume spike (>1.8x average)
# H4/L4 levels are tighter than H3/L3, increasing win rate while maintaining sufficient trades.
# Trend filter ensures alignment with daily momentum. Volume confirmation reduces false breakouts.
# Discrete sizing 0.25 to minimize fee churn. Target: 75-200 trades over 4 years.
# Primary timeframe: 4h, HTF: 1d for Camarilla levels and EMA50.

name = "4h_Camarilla_H4_L4_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels H4 and L4 from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla H4 and L4 levels
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (they update daily)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.8x 20-period average (balanced to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above H4 AND price > 1d EMA50 AND volume spike
            if (close[i] > camarilla_h4_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L4 AND price < 1d EMA50 AND volume spike
            elif (close[i] < camarilla_l4_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below L4 OR price < 1d EMA50
            if close[i] < camarilla_l4_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above H4 OR price > 1d EMA50
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals