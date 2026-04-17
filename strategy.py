#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h/1d trend alignment + Camarilla pivot breakout + volume filter.
Long when: price > 4h EMA50 AND price > 1d EMA200 (bullish alignment) AND price breaks above 4h Camarilla R1 with volume > 1.5x 20-period average.
Short when: price < 4h EMA50 AND price < 1d EMA200 (bearish alignment) AND price breaks below 4h Camarilla S1 with volume > 1.5x 20-period average.
Exit when price returns to 4h Camarilla midpoint (H4/L4) or reverses with volume.
Uses 4h/1d for trend direction and structure, 1h for precise entry timing.
Designed to capture medium-term breakouts with institutional volume while avoiding false signals in choppy or counter-trend markets.
Targets 15-35 trades/year by requiring HTF trend alignment + pivot breakout + volume confirmation.
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
    
    # Get 4h data for Camarilla levels and EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (based on prior 4h bar)
    range_4h = high_4h - low_4h
    r1_4h = close_4h + 0.833 * range_4h
    s1_4h = close_4h - 0.833 * range_4h
    midpoint_4h = close_4h  # Camarilla midpoint is close
    
    # Calculate 4h EMA50 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for EMA200 long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for long-term trend
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    midpoint_4h_aligned = align_htf_to_ltf(prices, df_4h, midpoint_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Align 1d EMA200 to 1h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or 
            np.isnan(midpoint_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend alignment: 4h EMA50 and 1d EMA200 agreement
        bullish_alignment = close[i] > ema50_4h_aligned[i] and close[i] > ema200_1d_aligned[i]
        bearish_alignment = close[i] < ema50_4h_aligned[i] and close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R1 with volume and bullish alignment
            if (close[i] > r1_4h_aligned[i] and 
                volume_confirmed and 
                bullish_alignment):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla S1 with volume and bearish alignment
            elif (close[i] < s1_4h_aligned[i] and 
                  volume_confirmed and 
                  bearish_alignment):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below S1 with volume (reversal)
            if (close[i] <= midpoint_4h_aligned[i] or 
                (close[i] < s1_4h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above R1 with volume (reversal)
            if (close[i] >= midpoint_4h_aligned[i] or 
                (close[i] > r1_4h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1dTrend_Camarilla_R1S1_Breakout_Volume"
timeframe = "1h"
leverage = 1.0