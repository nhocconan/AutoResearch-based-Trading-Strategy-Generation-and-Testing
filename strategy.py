#!/usr/bin/env python3
"""
Experiment #4697: 4h Donchian(20) Breakout + 1d Volume Spike + Chop Regime Filter
HYPOTHESIS: 4h price breaking Donchian(20) channels with volume confirmation (>2.0x avg volume) and aligned with 1d chop regime (CHOP > 61.8 = range, < 38.2 = trending) captures momentum in trending markets and mean-reversion in ranging markets. Uses discrete position sizing (0.25) to minimize fee drag while maintaining statistical significance. Works in both bull (breakouts with volume) and bear (short breakdowns with volume) markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4697_4h_donchian20_1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Chopiness Index (CHOP) for regime detection ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Chopiness Index: CHOP = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
        # Avoid division by zero
        range_1d = hh_1d - ll_1d
        chop_1d = np.full(len(close_1d), np.nan)
        mask = (range_1d > 0) & ~np.isnan(tr_sum)
        chop_1d[mask] = 100 * np.log10(tr_sum[mask] / range_1d[mask]) / np.log10(14)
        
        # Align HTF CHOP to 4h timeframe
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) from prior 20 bars ===
    # Use prior 20 bars' high/low (shifted by 1 to avoid look-ahead)
    ph = np.concatenate([[np.nan] * 20, high[:-20]])  # prior 20 bars high
    pl = np.concatenate([[np.nan] * 20, low[:-20]])   # prior 20 bars low
    
    # Rolling max/min of prior 20 bars
    donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14)  # Donchian, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation for breakouts (>2.0x)
        vol_breakout = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions
        breakout_long = price > donchian_high[i] and vol_breakout
        breakout_short = price < donchian_low[i] and vol_breakout
        
        # 1d Chop regime filter: 
        # CHOP > 61.8 = ranging market (mean revert at Donchian bands)
        # CHOP < 38.2 = trending market (breakout continuation)
        chop_value = chop_1d_aligned[i]
        chop_trending = chop_value < 38.2   # Trending regime - favor breakouts
        chop_ranging = chop_value > 61.8    # Ranging regime - favor mean reversion
        
        # Final entry conditions: breakout + volume + regime filter
        # In trending markets: trade breakouts
        # In ranging markets: trade mean reversion (fade Donchian touches)
        if chop_trending:
            # Trending market: breakout continuation
            if breakout_long:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif breakout_short:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif chop_ranging:
            # Ranging market: mean reversion at Donchian bands
            # Long when price touches lower band and starts reversing up
            # Short when price touches upper band and starts reversing down
            if price <= donchian_low[i] * 1.001 and close[i] > open[i]:  # Touch lower band + bullish candle
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif price >= donchian_high[i] * 0.999 and close[i] < open[i]:  # Touch upper band + bearish candle
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Neutral chop (38.2-61.8): no clear regime, stay flat
            signals[i] = 0.0
    
    return signals