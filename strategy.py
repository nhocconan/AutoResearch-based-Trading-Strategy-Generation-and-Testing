#!/usr/bin/env python3
"""
Experiment #1953: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts capture momentum while 12h HMA(21) filters for higher-timeframe trend alignment.
Volume confirmation (>1.5x 20-period average) ensures institutional participation. ATR-based stoploss manages risk.
Works in bull markets via breakouts and bear markets via short breakdowns. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1953_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA(21) trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h close
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 11  # floor(21/2)
    sqrt_len = 5   # floor(sqrt(21))
    
    def wma(arr, period):
        if period <= 0:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1, dtype=np.float64)
        return np.convolve(arr, weights[::-1], mode='same') / weights.sum()
    
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_12h = wma(raw_hma, sqrt_len)
    
    # 12h trend: 1 if close > HMA, -1 otherwise
    trend_12h = np.where(close_12h > hma_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 4h Indicators: Donchian(20) channels and volume MA ===
    # Donchian upper/lower: highest high/lowest low of last 20 periods
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
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
    
    warmup = max(lookback, 20, 14)  # sufficient for Donchian(20), volume MA, ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: stoploss or Donchian opposite touch
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2 * ATR below entry
                if price <= entry_price - 2.0 * entry_atr:
                    exit_signal = True
                # Exit if price touches Donchian lower (mean reversion)
                elif price <= lowest_low[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2 * ATR above entry
                if price >= entry_price + 2.0 * entry_atr:
                    exit_signal = True
                # Exit if price touches Donchian upper (mean reversion)
                elif price >= highest_high[i]:
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
        # Require 12h HMA trend alignment
        trend_bias = trend_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 12h trend up
            if trend_bias > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 12h trend down
            elif trend_bias < 0 and price < lowest_low[i]:
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