#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d + volume confirmation + 1w trend filter
# Uses daily Camarilla levels (H3, L3, H4, L4) for mean reversion and breakout signals
# Weekly EMA200 determines trend: only take mean reversion when price < weekly EMA200 (bearish bias in 2025+)
# Volume confirmation requires current volume > 2.0x 20-period average to filter weak signals
# Works in bull/bear: weekly trend filter ensures we align with higher timeframe direction
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_1w_camarilla_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, L3, L4
    range_1d = high_1d - low_1d
    h4 = close_1d + range_1d * 1.1 / 2
    h3 = close_1d + range_1d * 1.1 / 4
    l3 = close_1d - range_1d * 1.1 / 4
    l4 = close_1d - range_1d * 1.1 / 2
    
    # Align 1d Camarilla levels to 6h timeframe (completed 1d bar only)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend direction
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
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
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < L3 (mean reversion failure) OR price > H4 (strong breakout - take profit)
            if close[i] < l3_aligned[i] or close[i] > h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > H3 (mean reversion failure) OR price < L4 (strong breakout - take profit)
            if close[i] > h3_aligned[i] or close[i] < l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla levels
            if volume_confirmed:
                # Mean reversion long: price < L3 AND above weekly EMA200 (bullish weekly bias)
                if close[i] < l3_aligned[i] and close[i] > ema_200_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Mean reversion short: price > H3 AND below weekly EMA200 (bearish weekly bias)
                elif close[i] > h3_aligned[i] and close[i] < ema_200_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                # Breakout long: price > H4 AND above weekly EMA200 (bullish breakout with trend)
                elif close[i] > h4_aligned[i] and close[i] > ema_200_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price < L4 AND below weekly EMA200 (bearish breakout with trend)
                elif close[i] < l4_aligned[i] and close[i] < ema_200_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals