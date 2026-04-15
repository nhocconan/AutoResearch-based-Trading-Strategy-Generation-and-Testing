#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume confirmation and 1w trend filter
# Designed for low trade frequency (target 12-37/year) with clear mean-reversion logic
# Works in both bull (reversion from overbought) and bear (reversion from oversold) markets
# Uses volume spike to confirm rejection at pivot levels and trend filter to avoid counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels from previous 12h bar
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    camarilla_high = close_12h + 1.1 * (high_12h - low_12h) / 2
    camarilla_low = close_12h - 1.1 * (high_12h - low_12h) / 2
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price rejects below camarilla low + downtrend + volume spike
        if (close[i] < camarilla_low_aligned[i] and 
            close[i] < ema50_1w_aligned[i] and 
            volume[i] > 1.8 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price rejects above camarilla high + uptrend + volume spike
        elif (close[i] > camarilla_high_aligned[i] and 
              close[i] > ema50_1w_aligned[i] and 
              volume[i] > 1.8 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: price returns to mean (previous close) or trend fails
        elif position == 1 and (close[i] > close_12h[i] or 
                                close[i] > ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] < close_12h[i] or 
                                 close[i] < ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_1dVolume_1wEMA_Reversal"
timeframe = "12h"
leverage = 1.0