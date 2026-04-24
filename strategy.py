#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout + 1d EMA34 trend + volume spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when close breaks above Camarilla H3 AND price > 1d EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when close breaks below Camarilla L3 AND price < 1d EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: Close back below/above Camarilla pivot level for mean reversion in ranging markets.
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot levels for structure, volume confirmation for participation,
  EMA34 trend filter to avoid counter-trend trades.
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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from prior 4h OHLC
    # Using prior completed 4h bar for calculation
    prev_high = high_4h[-1] if len(high_4h) > 0 else high_4h[0]
    prev_low = low_4h[-1] if len(low_4h) > 0 else low_4h[0]
    prev_close = close_4h[-1] if len(close_4h) > 0 else close_4h[0]
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla H3, L3, and pivot levels
    camarilla_h3 = pivot + (range_val * 1.1 / 4.0)
    camarilla_l3 = pivot - (range_val * 1.1 / 4.0)
    camarilla_pivot = pivot
    
    # Align Camarilla levels to 4h timeframe (constant until new 4h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, np.full(len(df_4h), camarilla_h3))
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, np.full(len(df_4h), camarilla_l3))
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, np.full(len(df_4h), camarilla_pivot))
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above Camarilla H3 AND price > 1d EMA34 (uptrend)
                if curr_close > camarilla_h3_aligned[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below Camarilla L3 AND price < 1d EMA34 (downtrend)
                elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price returns to pivot (mean reversion)
            if curr_close < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price returns to pivot (mean reversion)
            if curr_close > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0