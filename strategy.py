#!/usr/bin/env python3
"""
Experiment #3214: 1h Donchian Breakout + 4h/1d Trend + Volume Spike + Session Filter
HYPOTHESIS: 1h Donchian(20) breakouts capture short-term trends with strict filters to avoid overtrading.
4h EMA(50) and 1d EMA(200) provide multi-timeframe trend alignment. Volume spike (>2.0x 20-period average) confirms breakout strength.
Session filter (08-20 UTC) reduces noise during low-liquidity hours. Position size 0.20.
Target: 60-150 total trades over 4 years = 15-37/year for 1h. Uses 4h/1d for signal direction, 1h only for entry timing.
Designed to work in both bull (trend continuation) and bear (mean reversion from extremes) markets by using price channels and volatility filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3214_1h_donchian20_4h_1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) for performance
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for EMA(50) trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for EMA(200) trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, 200, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC only ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (ATR proxy using Donchian width)
                donchian_width = highest_high[i] - lowest_low[i]
                if price < highest_since_entry - 1.5 * donchian_width:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (ATR proxy using Donchian width)
                donchian_width = highest_high[i] - lowest_low[i]
                if price > lowest_since_entry + 1.5 * donchian_width:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Multi-timeframe trend filter: 
            # - Long only when price > 4h EMA(50) AND price > 1d EMA(200)
            # - Short only when price < 4h EMA(50) AND price < 1d EMA(200)
            price_vs_4h_ema = price - ema_4h_aligned[i]
            price_vs_1d_ema = price - ema_1d_aligned[i]
            
            # Long entry: price breaks above Donchian high with bullish trend on both timeframes
            if price > highest_high[i] and price_vs_4h_ema > 0 and price_vs_1d_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish trend on both timeframes
            elif price < lowest_low[i] and price_vs_4h_ema < 0 and price_vs_1d_ema < 0:
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