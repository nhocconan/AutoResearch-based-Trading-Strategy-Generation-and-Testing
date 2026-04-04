#!/usr/bin/env python3
"""
Experiment #5709: 4h Donchian(20) breakout + 1d EMA trend + volume confirmation + chop filter
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned 
with daily EMA50 trend (price above EMA50 = bullish, below = bearish) capture high-probability 
trend continuation moves. Added choppiness index filter (CHOP > 61.8 = ranging, avoid entries) 
to reduce whipsaws in sideways markets. Daily EMA50 provides adaptive trend filter that works 
in both bull and bear markets. Volume confirms breakout strength. ATR trailing stop (2.5x) 
manages risk. Discrete sizing (0.25) minimizes fee churn. Target: 19-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5709_4h_donchian20_1d_ema_vol_chop_v1"
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
    
    # === HTF: 1d data for EMA50 trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        daily_ema50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        daily_ema50 = np.full(len(df_1d), np.nan)
    
    # Align daily EMA50 to 4h timeframe
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # === HTF: 1w data for chop regime filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 14:
        # True Range for chop calculation
        tr1 = df_1d['high'] - df_1d['low']
        tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
        tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
        tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_1d[0] = tr1[0]
        atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
        
        # Calculate highest high and lowest low over 14 periods
        hh_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
        ll_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
        
        # Sum of ATR over 14 periods
        sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        range_14 = hh_1d - ll_1d
        
        # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / range14) / log10(14)
        # Avoid division by zero and log of zero
        chop_raw = np.where(range_14 > 0, sum_atr_14 / range_14, 1.0)
        chop_raw = np.where(chop_raw > 0, chop_raw, 1.0)
        chop = 100 * np.log10(chop_raw) / np.log10(14)
        chop = np.where(np.isnan(chop), 50.0, chop)  # neutral when undefined
    else:
        chop = np.full(len(df_1d), 50.0)  # neutral default
    
    # Align chop to 4h timeframe
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
    
    warmup = max(20, 20, 14, 50, 14)  # Donchian, volume avg, ATR, EMA50, chop
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(daily_ema50_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below daily EMA50 (trend change) OR chop > 61.8 (ranging)
                if price <= stop_price or price <= daily_ema50_aligned[i] or chop_aligned[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above daily EMA50 (trend change) OR chop > 61.8 (ranging)
                if price >= stop_price or price >= daily_ema50_aligned[i] or chop_aligned[i] > 61.8:
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
        
        # Daily EMA50 bias: long above EMA50, short below EMA50
        long_bias = price > daily_ema50_aligned[i]
        short_bias = price < daily_ema50_aligned[i]
        
        # Chop filter: only trade when NOT ranging (CHOP <= 61.8)
        not_ranging = chop_aligned[i] <= 61.8
        
        # Entry conditions: breakout in direction of daily EMA50 bias with volume and not ranging
        long_setup = breakout_up and volume_confirmed and long_bias and not_ranging
        short_setup = breakout_down and volume_confirmed and short_bias and not_ranging
        
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