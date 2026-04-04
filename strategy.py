#!/usr/bin/env python3
"""
Experiment #5068: 12h Donchian Breakout + Volume Spike + Chop Regime
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts with volume > 1.8x average and choppiness index < 38.2 (trending regime) capture sustained momentum with controlled frequency. Uses 1w HTF for regime alignment: only trade in direction of weekly trend (price > weekly EMA50 for longs, < for shorts). Designed for 12-37 trades/year on 12h timeframe to minimize fee drag. Works in bull markets (buy breakouts) and bear markets (sell breakdowns) by following weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5068_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Weekly EMA50 for trend bias ===
    if len(df_1w) >= 50:
        ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    else:
        weekly_ema50_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation (1.8x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: Choppiness Index (14) for regime filter ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll = hh14 - ll14
    chop = np.full(n, np.nan)
    mask = (hh_ll > 0) & ~np.isnan(tr_sum)
    chop[mask] = 100 * np.log10(tr_sum[mask] / hh_ll[mask]) / np.log10(14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 14, 50)  # Donchian, Vol MA, Chop, Weekly EMA50 warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop[i]) or np.isnan(weekly_ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposite signal or stoploss ---
        if in_position:
            # Long exit: price breaks below Donchian low OR chop > 61.8 (range) OR weekly trend changes
            if position_side > 0:
                if (price <= lowest_low[i]) or (chop[i] > 61.8) or (price < weekly_ema50_aligned[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            # Short exit: price breaks above Donchian high OR chop > 61.8 (range) OR weekly trend changes
            else:
                if (price >= highest_high[i]) or (chop[i] > 61.8) or (price > weekly_ema50_aligned[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.8x)
        vol_confirm = vol_ratio[i] > 1.8
        
        # Regime filter: chop < 38.2 for trending market
        trending = chop[i] < 38.2
        
        # Weekly trend bias from 1w HTF
        weekly_bullish = price > weekly_ema50_aligned[i]
        weekly_bearish = price < weekly_ema50_aligned[i]
        
        # Long: price breaks above Donchian high + trending + weekly bullish bias + volume
        # Short: price breaks below Donchian low + trending + weekly bearish bias + volume
        long_signal = (price >= highest_high[i]) and trending and weekly_bullish and vol_confirm
        short_signal = (price <= lowest_low[i]) and trending and weekly_bearish and vol_confirm
        
        if long_signal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals