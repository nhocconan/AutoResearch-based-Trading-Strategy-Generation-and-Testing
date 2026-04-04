#!/usr/bin/env python3
"""
Experiment #3479: 6h Donchian Breakout + 12h/1d Weekly Pivot Filter + Volume Spike
HYPOTHESIS: 6h Donchian(20) breakouts with volume confirmation and 12h/1d weekly pivot direction
capture medium-term momentum while minimizing trades. Weekly pivots (R4/S4) act as strong
support/resistance - breaks indicate institutional interest. Works in bull (continuation) and
bear (mean reversion from extremes) via price channels. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3479_6h_donchian20_12h_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 12h data for weekly pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate weekly pivot points from prior 12h week (5 periods = 5*12h = 60h ≈ 2.5 days)
    # Using prior 5 periods to approximate weekly (adjustable)
    lookback_week = 5
    highest_high_week = pd.Series(high_12h).rolling(window=lookback_week, min_periods=lookback_week).max().values
    lowest_low_week = pd.Series(low_12h).rolling(window=lookback_week, min_periods=lookback_week).min().values
    close_week = pd.Series(close_12h).rolling(window=lookback_week, min_periods=lookback_week).mean().values
    
    # Weekly pivot calculation (standard formula)
    pivot_week = (highest_high_week + lowest_low_week + close_week) / 3.0
    r1 = 2 * pivot_week - lowest_low_week
    s1 = 2 * pivot_week - highest_high_week
    r2 = pivot_week + (highest_high_week - lowest_low_week)
    s2 = pivot_week - (highest_high_week - lowest_low_week)
    r3 = highest_high_week + 2 * (pivot_week - lowest_low_week)
    s3 = lowest_low_week - 2 * (highest_high_week - pivot_week)
    r4 = r3 + (highest_high_week - lowest_low_week)  # Extended resistance
    s4 = s3 - (highest_high_week - lowest_low_week)  # Extended support
    
    # Align weekly pivot levels to 6h
    pivot_week_aligned = align_htf_to_ltf(prices, df_12h, pivot_week)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 6h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_6h = 20
    highest_high_6h = pd.Series(high).rolling(window=lookback_6h, min_periods=lookback_6h).max().values
    lowest_low_6h = pd.Series(low).rolling(window=lookback_6h, min_periods=lookback_6h).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
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
    
    warmup = max(50, lookback_6h, lookback_week, 20, 14, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 6h Donchian channel (mean reversion)
                elif price <= highest_high_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 6h Donchian channel (mean reversion)
                elif price >= lowest_low_6h[i]:
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
            # 6h Donchian breakout with weekly pivot and 1d trend alignment
            price_vs_6h_high = price - highest_high_6h[i]
            price_vs_6h_low = price - lowest_low_6h[i]
            price_vs_r4 = price - r4_aligned[i]
            price_vs_s4 = price - s4_aligned[i]
            price_vs_ema = price - ema_1d_aligned[i]
            
            # Long entry: price breaks above 6h Donchian high AND above weekly R4 AND above 1d EMA
            if (price_vs_6h_high > 0 and 
                price_vs_r4 > 0 and 
                price_vs_ema > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 6h Donchian low AND below weekly S4 AND below 1d EMA
            elif (price_vs_6h_low < 0 and 
                  price_vs_s4 < 0 and 
                  price_vs_ema < 0):
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