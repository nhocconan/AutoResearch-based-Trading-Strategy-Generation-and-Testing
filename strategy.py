#!/usr/bin/env python3
"""
Experiment #5542: 12h Donchian(20) breakout + 1d EMA50 + volume confirmation + chop filter
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts with volume > 1.8x average and aligned with 1d EMA50 direction capture high-probability trend continuation moves. Adding choppiness index (CHOP) regime filter avoids whipsaws in ranging markets. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5542_12h_donchian20_1d_ema_vol_chop_v1"
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
    
    # === HTF: 1d data for EMA50 ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        trend_1d = np.where(df_1d['close'].values >= ema_1d, 1, -1)
        trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    else:
        trend_1d_aligned = np.full(n, 0)
    
    # === HTF: 1d data for Chopiness Index (CHOP) ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        
        highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        sum_atr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
        chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)  # avoid division by zero
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # neutral chop value
    
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
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 50, 14, 14)  # Donchian, volume avg, EMA warmup, ATR warmup, chop warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Donchian lower band break OR trend reversal OR chop > 61.8 (range)
                if price <= stop_price or price <= donchian_low[i] or (i < len(trend_1d_aligned) and trend_1d_aligned[i] == -1) or chop_aligned[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Donchian upper band break OR trend reversal OR chop > 61.8 (range)
                if price >= stop_price or price >= donchian_high[i] or (i < len(trend_1d_aligned) and trend_1d_aligned[i] == 1) or chop_aligned[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8
        chop_filter = chop_aligned[i] < 61.8  # only trade when not in choppy regime
        
        # Only trade in direction of 1d EMA50 trend
        long_entry = breakout_up and volume_confirmed and chop_filter and (i < len(trend_1d_aligned) and trend_1d_aligned[i] == 1)
        short_entry = breakout_down and volume_confirmed and chop_filter and (i < len(trend_1d_aligned) and trend_1d_aligned[i] == -1)
        
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