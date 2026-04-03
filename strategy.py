#!/usr/bin/env python3
"""
Experiment #2131: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture intermediate-term momentum, filtered by 1d weekly pivot 
direction (trend bias from higher timeframe structure) and volume confirmation to avoid false breaks.
Weekly pivot levels (calculated from prior 1d week) provide strong support/resistance zones.
Only trade breakouts in direction of weekly pivot trend to improve win rate.
Designed to work in both bull (trend following) and bear (mean reversion at extremes) markets.
Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2131_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week's OHLC (using 1d data)
    # Week = 5 trading days (approximate, ignoring weekends for simplicity)
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values  # last close of week
    
    # Weekly pivot point (PP) = (High + Low + Close) / 3
    pp = (week_high + week_low + week_close) / 3.0
    # Weekly R1 = 2*PP - Low
    r1 = 2 * pp - week_low
    # Weekly S1 = 2*PP - High
    s1 = 2 * pp - week_high
    # Weekly R2 = PP + (High - Low)
    r2 = pp + (week_high - week_low)
    # Weekly S2 = PP - (High - Low)
    s2 = pp - (week_high - week_low)
    # Weekly R3 = High + 2*(PP - Low)
    r3 = week_high + 2 * (pp - week_low)
    # Weekly S3 = Low - 2*(High - PP)
    s3 = week_low - 2 * (week_high - pp)
    
    # Trend bias: 1 if price above weekly PP, -1 if below (using prior week's values)
    # Align to 6h timeframe
    trend_bias_1d = np.where(close_1d > pp, 1, -1)
    trend_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # === 6h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection (moderate threshold to balance trades)
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
    SIZE = 0.25  # 25% position size
    
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
            np.isnan(trend_bias_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
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
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
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
        # Require weekly pivot trend alignment for bias filter
        trend_bias = trend_bias_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND weekly trend up
            if trend_bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND weekly trend down
            elif trend_bias < 0 and price < donchian_lower[i]:
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