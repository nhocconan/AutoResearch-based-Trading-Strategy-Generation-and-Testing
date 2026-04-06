#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversion with Daily Volume Spike and Weekly Trend Filter
Hypothesis: Price reverts to Camarilla pivot levels (H3/L3) with volume confirmation in trending markets.
Uses 1d Camarilla levels for mean reversion entries, filtered by 1w trend (EMA50) and volume spike (>2x average).
Works in bull (buy at L3 in uptrend) and bear (sell at H3 in downtrend). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # H4, H3, L3, L4 (we use H3/L3 for entries)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # EMA50 on weekly close
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 48) / 50
    
    # Weekly trend: above EMA50 = bullish, below = bearish
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    
    # Align weekly trend to 12h timeframe
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get daily data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-day average volume
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for EMA50 and volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 12h volume > 2x daily average volume (scaled)
        # Scale daily volume to 12h: approx 1/2 of daily volume (since 2x 12h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 2.0 * 2.0
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches H3 (take profit) OR against weekly trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] >= camarilla_h3_aligned[i] or
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price reaches L3 (take profit) OR against weekly trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] <= camarilla_l3_aligned[i] or
                trend_1w_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 24 bars flat (2 days)
            if bars_since_entry >= 24:
                # Mean reversion entries: touch L3/H3 with volume and trend alignment
                touches_l3 = low[i] <= camarilla_l3_aligned[i] * 1.001  # small buffer
                touches_h3 = high[i] >= camarilla_h3_aligned[i] * 0.999
                
                # Long: touch L3 in uptrend + volume
                if touches_l3 and trend_1w_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: touch H3 in downtrend + volume
                elif touches_h3 and trend_1w_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals