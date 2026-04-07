#!/usr/bin/env python3
"""
12h Camarilla Pivot with Daily Trend Filter and Volume Confirmation
Based on proven patterns: Camarilla pivot levels from 1d + volume spike + choppiness regime.
Uses 12h timeframe for lower trade frequency and 1d for pivot calculation.
Hypothesis: Price tends to revert to mean around Camarilla pivot levels in ranging markets,
and breakouts occur with volume confirmation in trending markets.
Works in both bull and bear via daily trend filter.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY PIVOT CALCULATION (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formulas:
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate ranges
    daily_range = daily_high - daily_low
    
    # Calculate Camarilla levels
    H4 = daily_close + 1.5 * daily_range
    H3 = daily_close + 1.0 * daily_range
    H2 = daily_close + 0.5 * daily_range
    H1 = daily_close + 0.25 * daily_range
    L1 = daily_close - 0.25 * daily_range
    L2 = daily_close - 0.5 * daily_range
    L3 = daily_close - 1.0 * daily_range
    L4 = daily_close - 1.5 * daily_range
    
    # Align levels to 12h timeframe (already shifted by 1 day in align_htf_to_ltf)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # === DAILY TREND FILTER (EMA50) ===
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    bull_trend = close > daily_ema_aligned  # Pre-compute for efficiency
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(H1_aligned[i]) or np.isnan(L1_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Current price levels
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 1:  # Long position
            # Exit: Price reaches H3 (take profit) OR breaks below L1 (stop)
            if curr_high >= H3_aligned[i] or curr_low <= L1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches L3 (take profit) OR breaks above H1 (stop)
            if curr_low <= L3_aligned[i] or curr_high >= H1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on daily trend
            if bull_trend[i]:
                # In bull market: look for long entries at support
                # Buy near L1-L2 with stop below L2, target at H1-H2
                if curr_low <= L1_aligned[i] and curr_close > L1_aligned[i]:
                    # Rejection of L1 level - potential long
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: look for short entries at resistance
                # Sell near H1-H2 with stop above H2, target at L1-L2
                if curr_high >= H1_aligned[i] and curr_close < H1_aligned[i]:
                    # Rejection of H1 level - potential short
                    position = -1
                    signals[i] = -0.25
    
    return signals