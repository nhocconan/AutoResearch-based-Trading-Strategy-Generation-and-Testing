#!/usr/bin/env python3
"""
Experiment #6187: 6h Donchian(20) breakout + 1d/1w pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with daily/weekly pivot levels capture medium-term momentum
in both bull and bear markets. Daily pivot (PP) defines trend bias, weekly pivot confirms structure.
Volume >1.8x average filters for institutional participation. ATR trailing stop manages risk.
Target: 75-200 trades over 4 years (19-50/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6187_6h_donchian20_1d_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for daily pivot ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate daily pivot from previous day's OHLC
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        pp = (prev_high + prev_low + prev_close) / 3.0  # Daily pivot point
        r1 = 2 * pp - prev_low  # Resistance 1
        s1 = 2 * pp - prev_high  # Support 1
        r2 = pp + (prev_high - prev_low)  # Resistance 2
        s2 = pp - (prev_high - prev_low)  # Support 2
        # Align to 6h timeframe (shifted by 1 day for completed bars only)
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for weekly pivot bias ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 2:
        # Calculate weekly pivot from previous week's OHLC
        prev_weekly_high = df_1w['high'].shift(1).values
        prev_weekly_low = df_1w['low'].shift(1).values
        prev_weekly_close = df_1w['close'].shift(1).values
        weekly_pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
        # Weekly bias: price above/below weekly pivot
        weekly_bias = np.where(close > weekly_pp, 1, np.where(close < weekly_pp, -1, 0))
        weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    else:
        weekly_bias_aligned = np.zeros(n)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14, 1) + 1  # Donchian, volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21:00-23:59 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below daily S1 (failed breakout)
                if price <= stop_price or price <= s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above daily R1 (failed breakout)
                if price >= stop_price or price >= r1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8  # Volume filter for stronger signals
        
        # Daily pivot bias: price relative to daily pivot point
        bullish_daily = price > pp_aligned[i]
        bearish_daily = price < pp_aligned[i]
        
        # Weekly pivot bias from 1w timeframe
        bullish_weekly = weekly_bias_aligned[i] > 0
        bearish_weekly = weekly_bias_aligned[i] < 0
        
        # Entry conditions: breakout with volume AND daily/weekly bias alignment
        # Long: breakout up with volume AND bullish daily bias AND bullish weekly bias
        # Short: breakout down with volume AND bearish daily bias AND bearish weekly bias
        long_entry = breakout_up and volume_confirmed and bullish_daily and bullish_weekly
        short_entry = breakout_down and volume_confirmed and bearish_daily and bearish_weekly
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals