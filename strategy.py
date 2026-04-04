#!/usr/bin/env python3
"""
Experiment #2316: 12h Donchian(20) breakout + 1d HMA trend + volume confirmation
HYPOTHESIS: 12h Donchian breakouts with 1d HMA trend filter and volume confirmation 
capture medium-term momentum while avoiding false breakouts. Works in bull markets 
(breakouts with trend) and bear markets (short breakdowns with downtrend).
- Primary: 12h Donchian channel breakout (20-bar)
- HTF: 1d HMA(21) trend filter (must align with breakout direction)
- Volume: Require > 1.8x 20-bar average spike to confirm participation
- Exit: Opposite Donchian channel touch or ATR(14) stop (2.5*ATR)
- Target: 50-150 total trades over 4 years (12-37/year) - suitable for 12h timeframe
- Position size: 0.25 (25%) to manage drawdown in volatile markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2316_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d HMA(21)
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values
        return hma_vals
    
    hma_1d = hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Donchian(20), ATR(14), Volume MA(20) ===
    # Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
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
                # Exit if price touches lower Donchian (contrarian exit)
                elif price <= lowest_20[i]:
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
                # Exit if price touches upper Donchian (contrarian exit)
                elif price >= highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d HMA trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]  # Either 1 or -1
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike and not np.isnan(trend_bias):
            # Long entry: price breaks above upper Donchian with uptrend
            if trend_bias > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian with downtrend
            elif trend_bias < 0 and price < lowest_20[i]:
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