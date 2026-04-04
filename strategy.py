#!/usr/bin/env python3
"""
Experiment #5723: 4h Donchian(20) breakout + 12h EMA(50) trend + volume confirmation + chop filter
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned with 12h EMA(50) direction capture high-probability trend continuation. 12h EMA provides intermediate trend filter that adapts to bull/bear markets. Chop filter (Choppiness Index > 61.8) avoids ranging markets where breakouts fail. Volume confirms breakout strength. ATR trailing stop (2.5x) manages risk. Discrete sizing (0.25) minimizes fee churn. Target: 19-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5723_4h_donchian20_12h_ema_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for EMA(50) trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 50:
        ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_12h = np.full(len(df_12h), np.nan)
    
    # Align 12h EMA to 4h timeframe (shifted by 1 for completed 12h bars only)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === HTF: 1d data for Choppiness Index regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_12h['high'].values if 'high' in df_12h.columns else df_1d['high'].values
        low_1d = df_12h['low'].values if 'low' in df_12h.columns else df_1d['low'].values
        close_1d = df_12h['close'].values if 'close' in df_12h.columns else df_1d['close'].values
        # True range for 1d
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_1d[0] = tr1[0]
        atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
        # Sum of ATR over 14 periods
        sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        # Max high - min low over 14 periods
        max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        range_1d = max_high_1d - min_low_1d
        # Choppiness Index: 100 * log10(sum_atr / range) / log10(period)
        chop = 100 * np.log10(sum_atr_1d / np.where(range_1d > 0, range_1d, 1)) / np.log10(14)
        # Replace inf/-inf with NaN
        chop = np.where(np.isfinite(chop), chop, np.nan)
    else:
        chop = np.full(len(df_1d), np.nan)
    
    # Align daily Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14, 50, 14)  # Donchian, volume avg, ATR, EMA, chop
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below 12h EMA (trend change) OR chop too high (ranging)
                if price <= stop_price or price <= ema_12h_aligned[i] or chop_aligned[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above 12h EMA (trend change) OR chop too high (ranging)
                if price >= stop_price or price >= ema_12h_aligned[i] or chop_aligned[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0
        
        # 12h EMA bias: long above EMA, short below EMA
        long_bias = price > ema_12h_aligned[i]
        short_bias = price < ema_12h_aligned[i]
        
        # Chop filter: only enter when market is trending (chop <= 61.8)
        trending = chop_aligned[i] <= 61.8
        
        # Entry conditions: breakout in direction of 12h EMA with volume and trending market
        long_setup = breakout_up and volume_confirmed and long_bias and trending
        short_setup = breakout_down and volume_confirmed and short_bias and trending
        
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