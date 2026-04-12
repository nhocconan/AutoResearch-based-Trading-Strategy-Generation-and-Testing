#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v1
# Daily Camarilla pivot levels with weekly trend filter (Hull MA on weekly close).
# Long when daily close breaks above H4 and weekly trend is up; short when breaks below L4 and weekly trend down.
# Volume confirmation (volume > 1.5x 20-day average) to filter false breakouts.
# Designed for low frequency: ~10-20 trades/year to minimize fee drag.
# Works in bull markets via breakouts above H4 in uptrend; in bear markets via breakdowns below L4 in downtrend.
name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to daily timeframe
    h4_level = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get weekly data for trend filter (Hull Moving Average on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        wma_trend = np.ones(len(prices))  # neutral if no weekly data
    else:
        # Hull MA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(arr, n):
            if n < 1:
                return arr
            weights = np.arange(1, n + 1)
            return np.convolve(arr, weights, mode='full')[:len(arr)] * weights.sum() / (weights.sum() * len(arr))
        
        close_1w = df_1w['close'].values
        n = len(close_1w)
        half_n = max(1, n // 2)
        sqrt_n = max(1, int(np.sqrt(n)))
        
        wma_half = wma(close_1w, half_n)
        wma_full = wma(close_1w, n)
        wma_raw = 2 * wma_half - wma_full
        hull_ma = wma(wma_raw, sqrt_n)
        
        # Align weekly Hull MA to daily timeframe
        wma_trend = align_htf_to_ltf(prices, df_1w, hull_ma)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup for volume MA
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]) or np.isnan(wma_trend[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H4 AND weekly trend up (close > weekly Hull MA)
        if close[i] > h4_level[i] and close[i] > wma_trend[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L4 AND weekly trend down (close < weekly Hull MA)
        elif close[i] < l4_level[i] and close[i] < wma_trend[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout with volume
        elif close[i] < l4_level[i] and position == 1 and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif close[i] > h4_level[i] and position == -1 and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals