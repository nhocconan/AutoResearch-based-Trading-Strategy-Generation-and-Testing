#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1w trend filter and volume confirmation
# Uses weekly trend to avoid counter-trend trades, reducing false breakouts
# Camarilla levels provide precise support/resistance for entries
# Volume confirmation ensures institutional participation
# Target: 20-40 trades/year to minimize fee drag
# Works in bull (long breakouts) and bear (short breakdowns) via trend filter
name = "4h_Camarilla_Pivot_WeeklyTrend_Volume_v1"
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
    
    # Get weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA200 for long-term trend
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1)
    ph = np.roll(high_1d, 1)
    pl = np.roll(low_1d, 1)
    pc = np.roll(close_1d, 1)
    ph[0] = high_1d[0]
    pl[0] = low_1d[0]
    pc[0] = close_1d[0]
    
    # Camarilla levels
    R4 = pc + ((ph - pl) * 1.1 / 2)  # Resistance 4
    R3 = pc + ((ph - pl) * 1.1 / 4)  # Resistance 3
    S3 = pc - ((ph - pl) * 1.1 / 4)  # Support 3
    S4 = pc - ((ph - pl) * 1.1 / 2)  # Support 4
    
    # Align Camarilla levels to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 4h ATR for stop loss
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for weekly EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or \
           np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or \
           np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Volume filter: current volume > 1.8x average volume (30-period)
        if i >= 30:
            avg_volume = np.mean(volume[i-30:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.8 * avg_volume
        
        if position == 0:
            # Long: break above R3 with weekly uptrend and volume
            if high[i] > R3_aligned[i-1] and volume_filter and ema200_1w_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with weekly downtrend and volume
            elif low[i] < S3_aligned[i-1] and volume_filter and ema200_1w_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S3 or ATR stop
            if close[i] < S3_aligned[i] or close[i] < close[i-1] - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R3 or ATR stop
            if close[i] > R3_aligned[i] or close[i] > close[i-1] + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals