#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Reversal with 1w EMA200 Trend Filter and Volume Spike Confirmation.
- Primary timeframe: 6h to reduce trade frequency and fee drag.
- HTF: 1w EMA200 for major trend direction (bullish if close > EMA200, bearish if close < EMA200).
- Camarilla pivots calculated from 1d OHLC: H3, L3, H4, L4 levels.
- Entry: Long when price crosses above H3 with volume spike AND 1w EMA200 bullish.
         Short when price crosses below L3 with volume spike AND 1w EMA200 bearish.
         H3/L3 are strong intraday support/resistance where reversals often occur.
- Exit: Opposite Camarilla level (L3 for long, H3 for short) or ATR trailing stop.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
This strategy fades at H3/L3 in the direction of the weekly trend, avoiding counter-trend trades.
Volume spikes confirm institutional interest at these key levels. Works in both bull and bear markets
by only taking trades aligned with the 1w trend, using Camarilla reversals for entry timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    df_1w_close = df_1w['close'].values
    ema_1w_200 = pd.Series(df_1w_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Calculate ATR(20) for trailing stop
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume spike detector: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20, 2)  # Need enough bars for EMA200, ATR, and 1d data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_200_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Get 1d OHLC for Camarilla calculation (previous completed 1d bar)
        # We need the 1d bar that closed before the current 6h bar
        # Find the index of the last completed 1d bar
        # Since we're on 6h timeframe, 4 bars = 1 day
        idx_1d = i // 4  # Each 1d bar spans 4 of our 6h bars
        if idx_1d < 1 or idx_1d >= len(df_1d):
            # Not enough 1d data yet
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get previous completed 1d bar (idx_1d - 1) to avoid look-ahead
        prev_1d_idx = idx_1d - 1
        if prev_1d_idx < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Calculate Camarilla pivots from previous 1d bar
        ph = df_1d['high'].iloc[prev_1d_idx]
        pl = df_1d['low'].iloc[prev_1d_idx]
        pc = df_1d['close'].iloc[prev_1d_idx]
        rng = ph - pl
        
        if rng <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Camarilla levels
        h3 = pc + (rng * 1.1 / 4)
        l3 = pc - (rng * 1.1 / 4)
        h4 = pc + (rng * 1.1 / 2)
        l4 = pc - (rng * 1.1 / 2)
        
        if position == 0:
            # Check for entry signals with volume spike
            # Long: price crosses above H3 with volume spike AND 1w EMA200 bullish
            if (curr_close > h3 and low[i-1] <= h3 and  # crossed above H3
                volume_spike[i] and 
                curr_close > ema_1w_200_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: price crosses below L3 with volume spike AND 1w EMA200 bearish
            elif (curr_close < l3 and high[i-1] >= l3 and  # crossed below L3
                  volume_spike[i] and 
                  curr_close < ema_1w_200_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: 
            # 1. Price reaches L3 (opposite level)
            # 2. ATR trailing stop: price < highest_high - 2.5*ATR
            if (curr_close < l3 or 
                curr_close < highest_since_entry - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: 
            # 1. Price reaches H3 (opposite level)
            # 2. ATR trailing stop: price > lowest_low + 2.5*ATR
            if (curr_close > h3 or 
                curr_close > lowest_since_entry + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Reversal_1wEMA200_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0