#!/usr/bin/env python3
"""
12h Camarilla Pivot H3/L3 Breakout with 1d EMA50 Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels represent stronger support/resistance than R1/S1.
Breakout above H3 with 1d uptrend (EMA50) and volume spike captures strong bullish momentum.
Breakdown below L3 with 1d downtrend and volume spike captures strong bearish momentum.
Uses 12h timeframe with 1d HTF for trend filter. Targets 50-150 total trades over 4 years.
Works in both bull and bear markets: trend filter ensures we only trade with higher timeframe momentum,
while H3/L3 levels and volume confirmation reduce false breakouts. Discrete position sizing (0.25) minimizes fee churn.
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
    
    # Get 1d data for EMA50 trend and Camarilla levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close for trend
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d candle)
    # Pivot = (H + L + C) / 3
    # H3 = Pivot + (H - L) * 1.1 / 4
    # L3 = Pivot - (H - L) * 1.1 / 4
    df_1d = df_1d.copy()
    df_1d['pivot'] = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    df_1d['h3'] = df_1d['pivot'] + (df_1d['high'] - df_1d['low']) * 1.1 / 4
    df_1d['l3'] = df_1d['pivot'] - (df_1d['high'] - df_1d['low']) * 1.1 / 4
    
    pivot_1d = df_1d['pivot'].values
    h3_1d = df_1d['h3'].values
    l3_1d = df_1d['l3'].values
    
    # Align 1d levels to 12h timeframe (previous day's levels available after 1d close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 24-period volume MA for 12h volume confirmation (24 periods = 12 days of 12h data)
    vol_ma_24_12h = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24_12h[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_ma_24_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_1d_aligned[i]
        pivot_val = pivot_1d_aligned[i]
        h3_val = h3_1d_aligned[i]
        l3_val = l3_1d_aligned[i]
        vol_ma_12h = vol_ma_24_12h[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 24-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: Break above H3 AND price > EMA50 (uptrend) AND volume confirmation
            long_entry = (curr_high > h3_val and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below L3 AND price < EMA50 (downtrend) AND volume confirmation
            short_entry = (curr_low < l3_val and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Price crosses below pivot OR EMA50 trend turns down
            if (curr_close < pivot_val or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price crosses above pivot OR EMA50 trend turns up
            if (curr_close > pivot_val or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0