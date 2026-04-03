#!/usr/bin/env python3
"""
Experiment #1963: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation
HYPOTHESIS: Donchian channel breakouts capture institutional order flow. 
- Primary signal: 4h price breaks above/below 20-period Donchian channel
- Trend filter: 12h HMA(21) alignment (avoid counter-trend trades)
- Volume confirmation: require volume > 1.8x 20-period average to filter weak breakouts
- Exit: ATR-based stoploss (2*ATR) or opposite Donchian channel touch
- Works in bull/bear markets by following HTF trend + volume-confirmed breakouts
- Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1963_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
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
    
    # Calculate 12h HMA(21): Hull Moving Average
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(arr, period):
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights, 'valid') / weights.sum()
    
    hma_12h = np.full(len(close_12h), np.nan)
    half_period = 21 // 2
    sqrt_period = int(np.sqrt(21))
    
    for i in range(20, len(close_12h)):
        wma_half = wma(close_12h[i-20:i+1], half_period)
        wma_full = wma(close_12h[i-20:i+1], 21)
        raw_hma = 2 * wma_half - wma_full
        hma_12h[i] = wma(raw_hma[-sqrt_period:], sqrt_period)[-1] if len(raw_hma) >= sqrt_period else np.nan
    
    # 12h HMA trend: 1 if price > HMA, -1 if price < HMA
    trend_12h = np.where(close_12h > hma_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 4h Indicators: Donchian Channel (20) and Volume MA ===
    # Donchian Channel: upper = max(high, 20), lower = min(low, 20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), volume MA, ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: price < entry_price - 2*ATR_at_entry
                if price < entry_price - 2.0 * entry_atr:
                    exit_signal = True
                # Exit if price touches lower Donchian (mean reversion)
                elif price <= low_roll_min[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: price > entry_price + 2*ATR_at_entry
                if price > entry_price + 2.0 * entry_atr:
                    exit_signal = True
                # Exit if price touches upper Donchian (mean reversion)
                elif price >= high_roll_max[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 12h HMA trend alignment for bias filter
        trend_bias = trend_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND 12h trend up
            if trend_bias > 0 and price > high_roll_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND 12h trend down
            elif trend_bias < 0 and price < low_roll_min[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals