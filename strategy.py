#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_reversal
# Hypothesis: Price rejection at Camarilla pivot levels (H3/L3) on 12h with volume confirmation and daily trend filter.
# Long when price closes below L3 then reverses above L3 with volume spike in uptrend (price > 1d EMA200).
# Short when price closes above H3 then reverses below H3 with volume spike in downtrend (price < 1d EMA200).
# Exit when price reaches opposite H3/L3 level or crosses 1d EMA200.
# Designed to capture mean-reversion bounces at key levels in ranging markets and pullbacks in trends.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_reversal"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    # Using previous day's values (shifted by 1)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day will have invalid values (from roll) but will be handled by min_periods in alignment
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 level or crosses below EMA200
            if close[i] >= camarilla_h3_aligned[i] or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 level or crosses above EMA200
            if close[i] <= camarilla_l3_aligned[i] or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.8x average volume
            volume_ok = volume[i] > 1.8 * avg_volume[i]
            
            # Reversal entries: rejection at L3 (long) or H3 (short)
            # Long: price was below L3 (previous close) and now closes above L3 with volume
            # Short: price was above H3 (previous close) and now closes below H3 with volume
            if i > 0:
                prev_close = close[i-1]
                if (prev_close < camarilla_l3_aligned[i] and close[i] > camarilla_l3_aligned[i] and 
                    close[i] > ema_200_1d_aligned[i] and volume_ok):
                    position = 1
                    signals[i] = 0.25
                elif (prev_close > camarilla_h3_aligned[i] and close[i] < camarilla_h3_aligned[i] and 
                      close[i] < ema_200_1d_aligned[i] and volume_ok):
                    position = -1
                    signals[i] = -0.25
    
    return signals