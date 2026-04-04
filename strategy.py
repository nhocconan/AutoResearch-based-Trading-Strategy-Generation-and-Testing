#!/usr/bin/env python3
"""
Experiment #5788: 12h Donchian(20) breakout + 1w/1d regime filter + volume confirmation
HYPOTHESIS: 12h Donchian breakouts aligned with weekly/daily regime (price above/below weekly pivot and daily EMA200) capture strong continuation moves with volume confirmation. Uses weekly pivot for long-term trend and daily EMA200 for medium-term filter to adapt to bull/bear/ranging markets. Targets 50-150 trades over 4 years with discrete sizing 0.25 to minimize fee drag. ATR-based trailing stop manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5788_12h_donchian20_1w_1d_regime_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for EMA200 and weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 200:
        # Daily EMA200 for medium-term trend filter
        ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    else:
        ema200_1d = np.full(len(df_1d), np.nan)
    
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week's OHLC (using 5-day approximate week)
        dh_5d = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
        dl_5d = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
        dc_5d = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
        # Weekly pivot = (Prior week High + Low + Close) / 3
        weekly_pivot = (dh_5d + dl_5d + dc_5d) / 3.0
        # Weekly R1/S1 = (2*Pivot) - Low, (2*Pivot) - High
        weekly_r1 = 2 * weekly_pivot - dl_5d
        weekly_s1 = 2 * weekly_pivot - dh_5d
    else:
        weekly_pivot = np.full(len(df_1d), np.nan)
        weekly_r1 = np.full(len(df_1d), np.nan)
        weekly_s1 = np.full(len(df_1d), np.nan)
    
    # Align 1d indicators to 12h timeframe (shifted by 1 for completed 1d bars only)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # === HTF: 1w data for additional regime filter (optional) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        # Weekly EMA50 for long-term trend filter
        ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema50_1w = np.full(len(df_1w), np.nan)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 12h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 200, 50, 5)  # Donchian, volume avg, ATR, EMA200, EMA50, 5d pivot warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        # Regime filters:
        # 1. Price above/below weekly pivot and R1/S1 levels (long-term)
        regime_long_ltf = price > weekly_pivot_aligned[i] and price > weekly_r1_aligned[i]
        regime_short_ltf = price < weekly_pivot_aligned[i] and price < weekly_s1_aligned[i]
        # 2. Price above/below daily EMA200 (medium-term)
        regime_long_mtf = price > ema200_1d_aligned[i]
        regime_short_mtf = price < ema200_1d_aligned[i]
        # 3. Price above/below weekly EMA50 (additional long-term filter)
        regime_long_htf = price > ema50_1w_aligned[i]
        regime_short_htf = price < ema50_1w_aligned[i]
        
        # Entry conditions: breakout in direction of ALL regime filters with volume confirmation
        long_setup = breakout_up and regime_long_ltf and regime_long_mtf and regime_long_htf and volume_confirmed
        short_setup = breakout_down and regime_short_ltf and regime_short_mtf and regime_short_htf and volume_confirmed
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals