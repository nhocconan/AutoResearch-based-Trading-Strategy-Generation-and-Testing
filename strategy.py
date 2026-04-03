#!/usr/bin/env python3
"""
Experiment #1984: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian channel breakouts capture institutional accumulation/distribution phases. 
Weekly HMA(21) filters for primary trend alignment to avoid counter-trend whipsaws. 
Volume confirmation (>1.5x 20-day average) ensures breakout authenticity. 
ATR-based stoploss (2.5x ATR(14)) limits drawdown. 
Works in bull markets via breakout continuation and in bear markets via breakdown continuation.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1984_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w HMA(21): Weighted Moving Average of WMA
    def wma(arr, period):
        weights = np.arange(1, period + 1, dtype=np.float64)
        return np.convolve(arr, weights / weights.sum(), mode='same')
    
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = wma(close_1w, half_n)
    wma_full = wma(close_1w, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21 = wma(raw_hma, sqrt_n)
    
    # 1w trend: price above/below HMA
    trend_1w = np.where(close_1w > hma_21, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators ===
    # Donchian Channel (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
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
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), volume MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss: 2.5 * ATR against position
            stoploss_hit = False
            if position_side > 0:  # Long
                if price < entry_price - 2.5 * entry_atr:
                    stoploss_hit = True
            else:  # Short
                if price > entry_price + 2.5 * entry_atr:
                    stoploss_hit = True
            
            # Exit conditions
            exit_signal = False
            
            if stoploss_hot:
                exit_signal = True
            elif position_side > 0:  # Long
                # Exit if price breaks below Donchian low (trend reversal)
                if price < donchian_low[i]:
                    exit_signal = True
            else:  # Short
                # Exit if price breaks above Donchian high (trend reversal)
                if price > donchian_high[i]:
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
        # Require 1w trend alignment for bias filter
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND 1w trend up
            if trend_bias > 0 and price > donchian_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND 1w trend down
            elif trend_bias < 0 and price < donchian_low[i]:
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