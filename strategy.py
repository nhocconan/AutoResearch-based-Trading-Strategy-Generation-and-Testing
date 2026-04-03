#!/usr/bin/env python3
"""
Experiment #1831: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot bias (price above/below weekly pivot) and volume confirmation (>1.8x average) capture medium-term swings with tight entries. Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Target: 75-150 total trades over 4 years (19-37/year) by requiring confluence of breakout, pivot alignment, and volume spike. Position size fixed at 0.25 to manage drawdown in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1831_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # === Weekly Pivot Points (using prior week's OHLC) ===
    # We need to group daily data into weeks, but to avoid look-ahead,
    # we'll use the prior completed week's OHLC for current week's pivot
    # Since we don't have explicit week grouping in the helper,
    # we'll approximate using rolling weekly lookback (5 trading days)
    # but shift by 5 to use prior week only
    if len(high_1d) >= 5:
        # Prior week's OHLC (5 trading days ago to avoid look-ahead)
        prior_week_high = np.maximum.reduce([np.roll(high_1d, 5*i) for i in range(1, 6) if 5*i < len(high_1d)])
        prior_week_low = np.minimum.reduce([np.roll(low_1d, 5*i) for i in range(1, 6) if 5*i < len(high_1d)])
        prior_week_close = np.roll(close_1d, 5)
        
        # Handle edge cases where roll creates invalid values
        prior_week_high = np.where(np.arange(len(high_1d)) >= 5, prior_week_high, high_1d)
        prior_week_low = np.where(np.arange(len(high_1d)) >= 5, prior_week_low, low_1d)
        prior_week_close = np.where(np.arange(len(high_1d)) >= 5, prior_week_close, close_1d)
        
        # Weekly pivot calculation: P = (H + L + C) / 3
        weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
        # Weekly support/resistance levels
        weekly_r1 = 2 * weekly_pivot - prior_week_low
        weekly_s1 = 2 * weekly_pivot - prior_week_high
        weekly_r2 = weekly_pivot + (prior_week_high - prior_week_low)
        weekly_s2 = weekly_pivot - (prior_week_high - prior_week_low)
        weekly_r3 = weekly_r2 + (prior_week_high - prior_week_low)
        weekly_s3 = weekly_s2 - (prior_week_high - prior_week_low)
    else:
        # Not enough data, fallback to daily pivot
        weekly_pivot = (high_1d + low_1d + close_1d) / 3.0
        weekly_r1 = 2 * weekly_pivot - low_1d
        weekly_s1 = 2 * weekly_pivot - high_1d
        weekly_r2 = weekly_pivot + (high_1d - low_1d)
        weekly_s2 = weekly_pivot - (high_1d - low_1d)
        weekly_r3 = weekly_r2 + (high_1d - low_1d)
        weekly_s3 = weekly_s2 - (high_1d - low_1d)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed weekly bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # === 6h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or
            np.isnan(weekly_s3_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require price to be beyond weekly S3/R3 for bias (extreme levels)
        price_above_r3 = price > weekly_r3_aligned[i]
        price_below_s3 = price < weekly_s3_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long breakout: price breaks above Donchian high AND above weekly R3
            if price > donch_high[i] and price_above_r3:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short breakdown: price breaks below Donchian low AND below weekly S3
            elif price < donch_low[i] and price_below_s3:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals