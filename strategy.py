#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d EMA50 trend filter + volume confirmation
    # Uses 12h timeframe for entries, 1d for trend filter (HTF)
    # Camarilla pivot levels from previous 1d: long at L3, short at H3 with volume > 1.5x average
    # Only take trades in direction of 1d EMA50 trend. Discrete sizing 0.25.
    # Target: 12-30 trades/year per symbol (~50-120 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels from previous 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H3 = Pivot + Range * 1.1/4
    # L3 = Pivot - Range * 1.1/4
    camarilla_pivot = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(1, len(df_1d)):
        # Use previous day's data to calculate today's pivots
        pivot = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        rng = high_1d[i-1] - low_1d[i-1]
        h3 = pivot + rng * 1.1 / 4.0
        l3 = pivot - rng * 1.1 / 4.0
        
        # Map 1d index to 12h indices: each 1d = 2*12h bars
        start_idx = i * 2
        end_idx = min(start_idx + 2, n)
        camarilla_pivot[start_idx:end_idx] = pivot
        camarilla_h3[start_idx:end_idx] = h3
        camarilla_l3[start_idx:end_idx] = l3
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above H3 in bullish trend
        if bullish_trend:
            long_entry = (close[i] > camarilla_h3[i]) and volume_spike[i]
        # Short breakout: price breaks below L3 in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < camarilla_l3[i]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < camarilla_l3[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > camarilla_h3[i]) or (not bullish_trend and not bearish_trend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "12h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0