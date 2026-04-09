#!/usr/bin/env python3
# 6h_camarilla_1d_trend_volume_v2
# Hypothesis: 6h Camarilla pivot levels filtered by 1d EMA50 trend and volume confirmation.
# In bull/bear markets, price tends to respect Camarilla levels (R3/S3, R4/S4) when aligned with higher timeframe trend.
# Volume confirmation ensures institutional participation. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_1d_trend_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's Camarilla levels (using prior completed day)
    # Calculate Camarilla for each completed day, then align to 6h bars
    camarilla_history = []
    for i in range(1, len(close_1d)):  # Start from 1 to have previous day
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        if range_val <= 0:
            camarilla_history.append({
                'R4': prev_close, 'R3': prev_close, 'S3': prev_close, 'S4': prev_close
            })
        else:
            camarilla_history.append({
                'R4': prev_close + range_val * 1.1/2,
                'R3': prev_close + range_val * 1.1/4,
                'S3': prev_close - range_val * 1.1/4,
                'S4': prev_close - range_val * 1.1/2
            })
    
    # Pad first day with NaN (no previous day)
    camarilla_history.insert(0, {
        'R4': np.nan, 'R3': np.nan, 'S3': np.nan, 'S4': np.nan
    })
    
    # Extract arrays and align to 6h timeframe
    R4_1d = np.array([x['R4'] for x in camarilla_history])
    R3_1d = np.array([x['R3'] for x in camarilla_history])
    S3_1d = np.array([x['S3'] for x in camarilla_history])
    S4_1d = np.array([x['S4'] for x in camarilla_history])
    
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below R3 OR trend turns bearish
            if close[i] < R3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 OR trend turns bullish
            if close[i] > S3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above R3 with bullish trend (continuation or bounce)
                if close[i] > R3_aligned[i] and close[i] > ema50_1d_aligned[i]:
                    # Additional filter: avoid buying too close to R4 (exhaustion)
                    if close[i] < R4_aligned[i] * 0.995:  # Not within 0.5% of R4
                        position = 1
                        signals[i] = 0.25
                # Short: price breaks below S3 with bearish trend (continuation or bounce)
                elif close[i] < S3_aligned[i] and close[i] < ema50_1d_aligned[i]:
                    # Additional filter: avoid selling too close to S4 (exhaustion)
                    if close[i] > S4_aligned[i] * 1.005:  # Not within 0.5% of S4
                        position = -1
                        signals[i] = -0.25
    
    return signals