#!/usr/bin/env python3
"""
Experiment #2287: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with weekly pivot bias and volume confirmation capture swing trades.
- Primary: 6h Donchian(20) breakout with volume > 2.0x 20-bar average (strict to limit trades)
- HTF: 1d weekly pivot (R4/S4 levels) - trade breakouts in direction of pivot bias
- Exit: ATR(14) trailing stop (2.5*ATR) or opposite Donchian touch
- Target: 50-150 total trades over 4 years (12-37/year) - optimized for 6h timeframe
- Weekly pivot provides structural bias: above weekly pivot = bullish, below = bearish
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2287_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week (using last 5 trading days)
    # For each day, we need prior week's H/L/C
    lookback = 5  # 5 trading days = 1 week
    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)
    pivot_close = np.full(n, np.nan)
    
    # Use rolling window on 1d data to get prior week's values
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    close_series = pd.Series(close_1d)
    
    # Prior week's high/low/close (shifted by 1 to avoid look-ahead)
    prior_week_high = high_series.rolling(window=lookback, min_periods=lookback).max().shift(1)
    prior_week_low = low_series.rolling(window=lookback, min_periods=lookback).min().shift(1)
    prior_week_close = close_series.rolling(window=lookback, min_periods=lookback).last().shift(1)
    
    pivot_high[:] = prior_week_high.values
    pivot_low[:] = prior_week_low.values
    pivot_close[:] = prior_week_close.values
    
    # Calculate weekly pivot point and support/resistance levels
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    pp = (pivot_high + pivot_low + pivot_close) / 3.0
    r1 = 2 * pp - pivot_low
    s1 = 2 * pp - pivot_high
    r2 = pp + (pivot_high - pivot_low)
    s2 = pp - (pivot_high - pivot_low)
    r3 = pivot_high + 2 * (pp - pivot_low)
    s3 = pivot_low - 2 * (pivot_high - pp)
    r4 = r3 + (pivot_high - pivot_low)
    s4 = s3 - (pivot_high - pivot_low)
    
    # Bias: above weekly pivot = bullish, below = bearish
    # We'll use R4/S4 as extreme levels for breakout confirmation
    bias_1d = np.where(close_1d > pp, 1, -1)  # 1 = bullish bias, -1 = bearish bias
    bias_1d_aligned = align_htf_to_ltf(prices, df_1d, bias_1d)
    
    # === 6h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - conservative for risk management
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(bias_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (mean reversion)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (mean reversion)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d weekly pivot bias for direction filter
        bias = bias_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average - very strict)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND bullish weekly bias
            if bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND bearish weekly bias
            elif bias < 0 and price < donchian_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals