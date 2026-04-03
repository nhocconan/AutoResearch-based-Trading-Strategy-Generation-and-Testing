#!/usr/bin/env python3
"""
Experiment #1876: 12h Donchian Breakout + Volume + Choppiness Regime
HYPOTHESIS: Donchian(20) breakouts capture strong momentum moves. Combined with volume confirmation (>1.5x average) and choppiness regime filter (CHOP > 61.8 = ranging, < 38.2 = trending), this strategy enters breakouts only in trending markets with volume support. Uses 1d HTF for trend alignment. Discrete position sizing of 0.25 minimizes fee churn. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1876_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend alignment (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Donchian Channel (20) ===
    # Upper band: highest high over 20 periods
    # Lower band: lowest low over 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: Choppiness Index (14) ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hl_range = hh_14 - ll_14
    chop = np.full(n, 50.0)  # default to neutral
    mask = (hl_range > 0) & (tr_sum > 0)
    chop[mask] = 100 * np.log10(tr_sum[mask] / hl_range[mask]) / np.log10(14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for Donchian(20) and CHOP(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop[i]) or
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or opposite signal ---
        if in_position:
            # ATR-based stoploss (using 20-period ATR approximation)
            # Approximate ATR as average true range over 20 periods
            atr_approx = tr_sum[i] / 20 if i >= 20 else 0
            if atr_approx > 0:
                if position_side > 0:  # Long position
                    if price < entry_price - 2.0 * atr_approx:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                else:  # Short position
                    if price > entry_price + 2.0 * atr_approx:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
            
            # Exit on opposite Donchian breakout
            if position_side > 0 and price < lowest_low[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and price > highest_high[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Require trending market (CHOP < 38.2 = strong trend)
        trending = chop[i] < 38.2
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if trending and volume_spike:
            # Long: price breaks above Donchian upper band
            if trend_bias > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short: price breaks below Donchian lower band
            elif trend_bias < 0 and price < lowest_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals