#!/usr/bin/env python3
"""
Experiment #3079: 6h Donchian Breakout + 12h HMA Trend + Volume Spike (Novel Adaptive)
HYPOTHESIS: 6h Donchian(20) breakouts with 12h HMA(21) trend filter and volume confirmation (>2.0x 20-period average) 
capture medium-term momentum while minimizing whipsaw. Adaptive position sizing based on trend strength (HMA slope) 
reduces exposure in choppy markets. ATR trailing stop (2.0x) manages risk. Designed for 6h timeframe to balance 
trade frequency and signal reliability in both bull (trend continuation) and bear (mean reversion from extremes) 
markets by using price channels and volatility filters. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3079_6h_donchian20_12h_hma_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h close
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_vals = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma_vals
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate HMA slope for adaptive sizing (trend strength)
    hma_slope = np.zeros_like(hma_12h_aligned)
    hma_slope[1:] = (hma_12h_aligned[1:] - hma_12h_aligned[:-1]) / hma_12h_aligned[:-1]
    hma_slope[0] = 0.0
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
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
    BASE_SIZE = 0.25  # Base position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback, 20, 14, 21)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    # Adaptive sizing: increase size in strong trends, reduce in weak
                    trend_strength = min(abs(hma_slope[i]) * 100, 1.0)  # cap at 1.0
                    adaptive_size = BASE_SIZE * (0.5 + 0.5 * trend_strength)  # range [0.5*BASE, BASE]
                    signals[i] = position_side * adaptive_size
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    # Adaptive sizing: increase size in strong trends, reduce in weak
                    trend_strength = min(abs(hma_slope[i]) * 100, 1.0)  # cap at 1.0
                    adaptive_size = BASE_SIZE * (0.5 + 0.5 * trend_strength)  # range [0.5*BASE, BASE]
                    signals[i] = position_side * adaptive_size
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # 12h HMA trend filter: only long above HMA, short below HMA
            price_vs_hma = price - hma_12h_aligned[i]
            
            # Long entry: price breaks above Donchian high with bullish 12h trend
            if price > highest_high[i] and price_vs_hma > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                # Adaptive sizing at entry
                trend_strength = min(abs(hma_slope[i]) * 100, 1.0)  # cap at 1.0
                adaptive_size = BASE_SIZE * (0.5 + 0.5 * trend_strength)  # range [0.5*BASE, BASE]
                signals[i] = adaptive_size
            # Short entry: price breaks below Donchian low with bearish 12h trend
            elif price < lowest_low[i] and price_vs_hma < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                # Adaptive sizing at entry
                trend_strength = min(abs(hma_slope[i]) * 100, 1.0)  # cap at 1.0
                adaptive_size = BASE_SIZE * (0.5 + 0.5 * trend_strength)  # range [0.5*BASE, BASE]
                signals[i] = -adaptive_size
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals