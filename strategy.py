#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot-based mean reversion with 1d volume filter and 1w trend filter
# Designed for low trade frequency (target 15-25/year) with high win probability
# Uses Camarilla levels (L3/L4 for short, H3/H4 for long) from daily pivots
# Only takes mean-reversion trades when 1w EMA50 trend aligns with expected reversal
# Volume spike required to confirm institutional interest at pivot levels
# Works in ranging markets (mean reversion) and trending markets (pullbacks to value)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Load 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Standard Camarilla: H4 = C + 1.1*(H-L)/2, H3 = C + 1.1*(H-L)/4
    # L3 = C - 1.1*(H-L)/4, L4 = C - 1.1*(H-L)/2
    camarilla_H4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_H3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_L3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_L4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Volume average (20-period on 1d) for spike detection
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price touches/slightly pierces L3/L4 + uptrend + volume spike
        if (low[i] <= camarilla_L3_aligned[i] * 1.002 and  # Allow 0.2% penetration
            close[i] > camarilla_L3_aligned[i] and          # Close back above L3
            close[i] > ema50_1w_aligned[i] and              # Above weekly trend
            volume[i] > 2.0 * vol_avg_aligned[i] and        # Strong volume confirmation
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches/slightly pierces H3/H4 + downtrend + volume spike
        elif (high[i] >= camarilla_H3_aligned[i] * 0.998 and  # Allow 0.2% penetration
              close[i] < camarilla_H3_aligned[i] and          # Close back below H3
              close[i] < ema50_1w_aligned[i] and              # Below weekly trend
              volume[i] > 2.0 * vol_avg_aligned[i] and        # Strong volume confirmation
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price returns to mean (pivot) or trend breaks
        elif position == 1 and (close[i] >= (camarilla_H3_aligned[i] + camarilla_L3_aligned[i]) / 2 or
                                close[i] < ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= (camarilla_H3_aligned[i] + camarilla_L3_aligned[i]) / 2 or
                                 close[i] > ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_Pivot_MeanReversion_1dVol_1wTrend"
timeframe = "4h"
leverage = 1.0