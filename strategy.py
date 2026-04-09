#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Uses 1d HMA(21) for higher timeframe trend direction (adapts to bull/bear)
# 12h Camarilla levels provide precise entry/exit structure
# Volume confirmation ensures breakout authenticity
# Discrete sizing 0.25 limits drawdown and reduces fee churn
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_1d_camarilla_hma_volume_v1"
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
    
    # Load 1d data ONCE before loop for HMA and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d HMA(21) for trend filter
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        wma_vals = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            wma_vals[i] = np.dot(values[i - window + 1:i + 1], weights) / weights.sum()
        return wma_vals
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    hma_1d = 2 * wma_half - wma_full
    hma_1d = wma(hma_1d, sqrt_len)
    
    # Align 1d HMA to 12h timeframe (wait for 1d bar close)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(n):
        if i < 1:
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Use previous day's OHLC (1d bar i-1)
            prev_high = df_1d['high'].iloc[i-1] if i-1 < len(df_1d) else np.nan
            prev_low = df_1d['low'].iloc[i-1] if i-1 < len(df_1d) else np.nan
            prev_close = df_1d['close'].iloc[i-1] if i-1 < len(df_1d) else np.nan
            
            if np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close):
                camarilla_h3[i] = np.nan
                camarilla_l3[i] = np.nan
                camarilla_h4[i] = np.nan
                camarilla_l4[i] = np.nan
            else:
                # Camarilla levels calculation
                range_val = prev_high - prev_low
                camarilla_h3[i] = prev_close + range_val * 1.1 / 4
                camarilla_l3[i] = prev_close - range_val * 1.1 / 4
                camarilla_h4[i] = prev_close + range_val * 1.1 / 2
                camarilla_l4[i] = prev_close - range_val * 1.1 / 2
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h4[i]) or 
            np.isnan(camarilla_l4[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR price < 1d HMA (trend change)
            if close[i] < camarilla_l3[i] or close[i] < hma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR price > 1d HMA (trend change)
            if close[i] > camarilla_h3[i] or close[i] > hma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla breakout + 1d HMA filter
            if volume_confirmed:
                # Long entry: price > Camarilla H3 AND price > 1d HMA (bullish alignment)
                if close[i] > camarilla_h3[i] and close[i] > hma_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Camarilla L3 AND price < 1d HMA (bearish alignment)
                elif close[i] < camarilla_l3[i] and close[i] < hma_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals