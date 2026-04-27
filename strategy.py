#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly pivot reversal with volume confirmation and ADX filter.
# Long when price crosses above weekly pivot from below with volume spike and ADX < 25 (range).
# Short when price crosses below weekly pivot from above with volume spike and ADX < 25.
# Uses weekly pivot points from prior week: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H.
# Designed for 10-30 trades/year per symbol (40-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by fading extremes in ranging conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot levels: P, R1, S1
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    
    # Align weekly pivot levels to daily timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # ADX filter: only trade in ranging markets (ADX < 25)
    # Calculate ADX(14) on daily data
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    period = 14
    atr = wilders_smooth(tr, period)
    plus_di = 100 * wilders_smooth(plus_dm, period) / atr
    minus_di = 100 * wilders_smooth(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price crosses above pivot from below AND ADX < 25 AND volume spike
        if (close[i] > pivot_aligned[i] and close[i-1] <= pivot_aligned[i-1] and
            adx[i] < 25 and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price crosses below pivot from above AND ADX < 25 AND volume spike
        elif (close[i] < pivot_aligned[i] and close[i-1] >= pivot_aligned[i-1] and
              adx[i] < 25 and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyPivotReversion_Volume_ADX"
timeframe = "1d"
leverage = 1.0