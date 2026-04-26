#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter
Hypothesis: Daily Camarilla pivot R1/S1 breakout with weekly EMA34 trend filter and volume confirmation (>1.5x 20-day MA).
Long when price breaks above R1 with weekly uptrend and volume spike.
Short when price breaks below S1 with weekly downtrend and volume filter.
Uses discrete position sizing (0.25) to minimize fee churn.
Designed to capture breakouts in trending markets while filtering false breakouts in ranging markets.
Target: 7-25 trades/year (30-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    uptrend_1w = close > ema_34_1w_aligned
    downtrend_1w = close < ema_34_1w_aligned
    
    # Calculate daily Camarilla pivot points (using previous day's OHLC)
    # Need to shift by 1 to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.1 / 12)
    S1 = pivot - (range_val * 1.1 / 12)
    
    # Volume confirmation: volume > 1.5x 20-day MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for weekly EMA + 1 for pivot shift + 20 for volume MA)
    start_idx = 55
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(pivot[i]) or np.isnan(R1[i]) or 
            np.isnan(S1[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with weekly uptrend and volume spike
            if close[i] > R1[i] and uptrend_1w[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with weekly downtrend and volume spike
            elif close[i] < S1[i] and downtrend_1w[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below pivot OR weekly trend changes to downtrend
            if close[i] < pivot[i] or not uptrend_1w[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above pivot OR weekly trend changes to uptrend
            if close[i] > pivot[i] or not downtrend_1w[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0