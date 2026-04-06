#!/usr/bin/env python3
"""
6h Bollinger Band Reversal with 12h Trend Filter and Volume Confirmation
Hypothesis: In mean-reverting markets, price reverses from Bollinger Band extremes. 
Filter by 12h EMA trend to avoid counter-trend trades and volume for confirmation.
Works in bull (buy dips below lower BB in uptrend) and bear (sell rallies above upper BB in downtrend).
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bbr_12h_trend_vol_v1"
timeframe = "6h"
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
    
    # 20-period Bollinger Bands (2 std dev)
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate basis (SMA)
    basis = np.full(n, np.nan)
    for i in range(bb_length, n):
        basis[i] = np.mean(close[i-bb_length:i])
    
    # Calculate standard deviation
    dev = np.full(n, np.nan)
    for i in range(bb_length, n):
        dev[i] = np.std(close[i-bb_length:i])
    
    # Upper and lower bands
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev
    
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
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA50 on 12h close
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 48) / 50
    
    # 12h trend: above EMA50 = bullish, below = bearish
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    
    # Align 12h trend to 6h timeframe
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Get 12h data for volume confirmation
    volume_12h = df_12h['volume'].values
    
    # 20-period average volume on 12h
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(50, bb_length)  # Need enough data for BB and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_12h_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(basis[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 12h average volume (scaled)
        # Scale 12h volume to 6h: approx 1/2 of 12h volume (since 2x 6h in 12h)
        vol_threshold = vol_ma_12h_aligned[i] / 2.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses above basis (mean reversion) OR against 12h trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] > basis[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price crosses below basis (mean reversion) OR against 12h trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] < basis[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries - mean reversion from BB extremes
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Mean reversion entries: price at BB bands with 12h trend
                at_lower = close[i] <= lower[i]
                at_upper = close[i] >= upper[i]
                
                # Long: price at/below lower BB with bullish 12h trend + volume
                if at_lower and trend_12h_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: price at/above upper BB with bearish 12h trend + volume
                elif at_upper and trend_12h_aligned[i] == -1 and volume_filter:
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