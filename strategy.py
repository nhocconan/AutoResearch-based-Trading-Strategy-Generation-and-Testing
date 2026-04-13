#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout
Hypothesis: Uses weekly Camarilla pivot levels on 1d timeframe to trade breakouts with volume confirmation.
Long when price breaks above H3 with volume > 1.5x average, short when breaks below L3.
Filters out low volatility periods using weekly ATR percentile to avoid false breakouts.
Designed for low-frequency, high-conviction trades (target: 10-25 trades/year).
Works in both bull and bear markets by trading institutional levels with volume validation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR (14) for volatility filter
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 52-week percentile of ATR for volatility regime (avoid low volatility)
    atr_series = pd.Series(atr_14)
    atr_percentile = atr_series.rolling(window=52, min_periods=26).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Avoid extremely low volatility (< 20th percentile) to prevent false breakouts
    vol_filter = atr_percentile >= 20.0
    
    # Calculate Camarilla pivot levels for each week
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    high_low = high_1w - low_1w
    h3 = close_1w + 1.1 * high_low
    l3 = close_1w - 1.1 * high_low
    h4 = close_1w + 1.5 * high_low
    l4 = close_1w - 1.5 * high_low
    
    # Align weekly levels to daily timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1w, vol_filter)
    
    # Daily volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready or volatility filter fails
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_filter_aligned[i]) or not vol_filter_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > h3_aligned[i] and volume_expansion[i]
        breakout_down = close[i] < l3_aligned[i] and volume_expansion[i]
        
        # Exit conditions: price returns to opposite side of pivot
        exit_long = position == 1 and close[i] < (h3_aligned[i] + l3_aligned[i]) / 2
        exit_short = position == -1 and close[i] > (h3_aligned[i] + l3_aligned[i]) / 2
        
        if breakout_up and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_down and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout"
timeframe = "1d"
leverage = 1.0