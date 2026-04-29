#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + weekly ATR filter + volume confirmation
# Donchian breakouts capture momentum in both bull and bear markets
# Weekly ATR filter avoids trading during low volatility regimes
# Volume confirmation ensures institutional participation
# Uses discrete position sizing (0.25) to minimize fee churn
# Target: 12-25 trades/year (50-100 total over 4 years)

name = "12h_Donchian20_Breakout_WeeklyATR_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR (14-period) for volatility filter
    # ATR = TR smoothed, where TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close_1w = df_1w['close'].shift(1).values
    tr1 = df_1w['high'].values - df_1w['low'].values
    tr2 = np.abs(df_1w['high'].values - prev_close_1w)
    tr3 = np.abs(df_1w['low'].values - prev_close_1w)
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ATR to 12h timeframe (completed weekly bar only)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.5 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 30)  # warmup for Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr_1w = atr_1w_aligned[i]
        
        # Skip if volatility too low (ATR below 25th percentile of recent values)
        # Simple filter: require ATR > 0.5 * 50-period median ATR
        if i >= 100:
            atr_median = np.nanmedian(atr_1w_aligned[max(0, i-50):i])
            if curr_atr_1w < 0.5 * atr_median:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
                continue
        
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above upper Donchian band + volume confirmation
            if curr_close > highest_high[i] and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian band + volume confirmation
            elif curr_close < lowest_low[i] and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when price breaks below lower Donchian band
            if curr_close < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when price breaks above upper Donchian band
            if curr_close > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals