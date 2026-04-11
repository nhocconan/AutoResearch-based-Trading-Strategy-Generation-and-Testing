#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: Daily Camarilla pivot breakout with weekly volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (L3/L3, H3/H3) act as strong support/resistance.
# Breakouts with weekly volume confirmation capture institutional moves.
# Weekly volume filter avoids false breakouts in low-volume environments.
# Works in bull/bear by trading breakouts in direction of weekly trend (using weekly EMA20).
# Designed for very low trade frequency (<25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily typical price for Camarilla calculation
    typical_price = (high + low + close) / 3.0
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly volume average (20-period) for confirmation
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after weekly EMA warmup
        # Need at least 2 days of data for Camarilla (yesterday's data)
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Use previous day's data for Camarilla calculation (no look-ahead)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels for today based on yesterday's range
        if prev_high == prev_low:  # Avoid division by zero
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
            
        range_val = prev_high - prev_low
        camarilla_mult = 1.1 / 12  # Standard Camarilla multiplier
        
        # Key levels: L3 (support), H3 (resistance)
        l3 = prev_close - range_val * camarilla_mult * 4
        h3 = prev_close + range_val * camarilla_mult * 4
        
        # Skip if any required data is invalid
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i]):
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current weekly volume > 20-period average
        # Use aligned weekly volume (current week's volume so far)
        vol_1w_series = df_1w['volume'].values
        if len(vol_1w_series) == 0:
            vol_confirm = False
        else:
            vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w_series)
            vol_confirm = (not np.isnan(vol_1w_aligned[i]) and 
                          vol_1w_aligned[i] > vol_avg_20_1w_aligned[i])
        
        # Trend filter: close vs weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above H3 AND uptrend AND volume confirmation
        if (not np.isnan(h3) and close[i] > h3 and uptrend and vol_confirm and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below L3 AND downtrend AND volume confirmation
        elif (not np.isnan(l3) and close[i] < l3 and downtrend and vol_confirm and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Price returns to yesterday's close (mean reversion to pivot)
        elif position == 1 and close[i] < prev_close:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > prev_close:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals