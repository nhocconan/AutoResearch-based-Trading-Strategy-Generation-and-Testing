#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 1d HTF for EMA34 trend alignment (proven pattern from DB)
- Camarilla levels calculated from previous completed 1d bar's OHLC (standard formula)
- Long when price breaks above R1 AND price > 1d EMA34 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below S1 AND price < 1d EMA34 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the Camarilla H3/L3 midpoint (mean reversion structure)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) as per 4h timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Shift by 1 to use previous completed 4h bar's OHLC for entry logic
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(open_price, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_open[0] = np.nan
    
    # Get 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous completed 1d bar's OHLC
    # Standard Camarilla formula based on previous day's range
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    H3 = np.full(n, np.nan)
    L3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous completed 1d bar's OHLC (we need to map 4h index to 1d index)
        # Since we're using align_htf_to_ltf later, we'll calculate on 1d then align
        pass  # Will calculate after getting 1d OHLC
    
    # Calculate Camarilla levels on 1d data then align to 4h
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous 1d bar's OHLC
    R1_1d = np.full(len(df_1d), np.nan)
    S1_1d = np.full(len(df_1d), np.nan)
    H3_1d = np.full(len(df_1d), np.nan)
    L3_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        range_val = high_prev - low_prev
        
        if range_val > 0:
            R1_1d[i] = close_prev + range_val * 1.1 / 12
            S1_1d[i] = close_prev - range_val * 1.1 / 12
            H3_1d[i] = close_prev + range_val * 1.1 / 4
            L3_1d[i] = close_prev - range_val * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R1 = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1 = align_htf_to_ltf(prices, df_1d, S1_1d)
    H3 = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3 = align_htf_to_ltf(prices, df_1d, L3_1d)
    
    # Midpoint for exit (between H3 and L3)
    midpoint = (H3 + L3) / 2.0
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 1d EMA34 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(midpoint[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND uptrend AND volume confirmation
            if close[i] > R1[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND downtrend AND volume confirmation
            elif close[i] < S1[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint (H3/L3 midpoint)
            if close[i] < midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint (H3/L3 midpoint)
            if close[i] > midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0