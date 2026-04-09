#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels with 1w HMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from daily data: long at L3, short at H3 with stop at L4/H4
# 1w HMA50 filter ensures trades align with weekly trend for stability in bull/bear markets
# Volume confirmation reduces false breakouts
# Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year)
# Works in bull/bear: HMA50 adapts to trend, Camarilla provides mean-reversion structure

name = "1d_1w_camarilla_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w HMA50 trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    half_n = int(50/2 + 0.5)
    wma_half = pd.Series(close_1w).rolling(window=half_n, min_periods=half_n).mean()
    wma_full = pd.Series(close_1w).rolling(window=50, min_periods=50).mean()
    hma_50_1w = (2 * wma_half - wma_full).values
    
    # Align 1w HMA50 to 1d timeframe
    hma_50_1d = align_htf_to_ltf(prices, df_1w, hma_50_1w)
    
    # Calculate 20-period average volume for volume confirmation (1d volume)
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
        if np.isnan(hma_50_1d[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels for today (using yesterday's OHLC)
        if i < 1:  # Need at least yesterday's data
            signals[i] = 0.0
            continue
            
        # Yesterday's OHLC for Camarilla calculation
        y_high = high[i-1]
        y_low = low[i-1]
        y_close = close[i-1]
        
        # Camarilla levels
        pivot = (y_high + y_low + y_close) / 3
        range_hl = y_high - y_low
        
        # Key levels: L3, H3 for entry; L4, H4 for stop
        l3 = pivot - (range_hl * 1.1 / 4)
        h3 = pivot + (range_hl * 1.1 / 4)
        l4 = pivot - (range_hl * 1.1 / 2)
        h4 = pivot + (range_hl * 1.1 / 2)
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (mean reversion complete) OR trend turns bearish
            if close[i] < l3 or close[i] < hma_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 (mean reversion complete) OR trend turns bullish
            if close[i] > h3 or close[i] > hma_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and trend filter
            if volume_confirm:
                # Long entry: price crosses above L3 AND price > 1w HMA50 (bullish weekly trend)
                if close[i] > l3 and close[i-1] <= l3 and close[i] > hma_50_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price crosses below H3 AND price < 1w HMA50 (bearish weekly trend)
                elif close[i] < h3 and close[i-1] >= h3 and close[i] < hma_50_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals