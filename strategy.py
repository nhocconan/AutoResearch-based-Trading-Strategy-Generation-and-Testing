#!/usr/bin/env python3
"""
Experiment #111: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation

HYPOTHESIS: Donchian(20) breakouts on 6h timeframe, filtered by 1d weekly pivot levels (R3/S3 for mean reversion, R4/S4 for breakout continuation) and 1d volume spike (>1.5x average), capture institutional participation in both trending and ranging markets. Weekly pivots act as dynamic support/resistance where price tends to reverse at R3/S3 and accelerate through R4/S4. Volume confirmation ensures breakouts have conviction. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels and volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week (using prior 5 trading days approx)
    # Weekly pivot = (PriorWeek High + PriorWeek Low + PriorWeek Close) / 3
    if len(df_1d) >= 5:
        # Shift by 5 to get prior week's OHLC (approximation for weekly)
        prior_week_high = df_1d['high'].shift(5).rolling(window=5, min_periods=5).max().values
        prior_week_low = df_1d['low'].shift(5).rolling(window=5, min_periods=5).min().values
        prior_week_close = df_1d['close'].shift(5).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot calculation
        pp = (prior_week_high + prior_week_low + prior_week_close) / 3.0
        
        # Weekly pivot levels
        r1 = 2 * pp - prior_week_low
        s1 = 2 * pp - prior_week_high
        r2 = pp + (prior_week_high - prior_week_low)
        s2 = pp - (prior_week_high - prior_week_low)
        r3 = prior_week_high + 2 * (pp - prior_week_low)
        s3 = prior_week_low - 2 * (prior_week_high - pp)
        r4 = prior_week_high + 3 * (pp - prior_week_low)
        s4 = prior_week_low - 3 * (prior_week_high - pp)
        
        # Align to 6h timeframe (shift(1) for completed weekly bars only)
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pp_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === HTF: 1d volume confirmation (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Donchian(20) channels
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Pivot Zone Logic ---
        # Mean reversion zone: between R3 and S3 (fade extremes)
        in_mean_reversion_zone = (close[i] <= r3_aligned[i]) and (close[i] >= s3_aligned[i])
        
        # Breakout zone: beyond R4 or S4 (continuation)
        is_breakout_long = close[i] > r4_aligned[i]
        is_breakout_short = close[i] < s4_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) on 1d ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite pivot level or Donchian low
                if close[i] <= donchian_low[i] or close[i] <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite pivot level or Donchian high
                if close[i] >= donchian_high[i] or close[i] >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout with volume and pivot alignment
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            (is_breakout_long or in_mean_reversion_zone)
        )
        
        # Short: Donchian breakdown with volume and pivot alignment
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            (is_breakout_short or in_mean_reversion_zone)
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals