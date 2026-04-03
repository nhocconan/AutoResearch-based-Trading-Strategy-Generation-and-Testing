#!/usr/bin/env python3
"""
Experiment #1908: 12h Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: 12h Donchian(20) breakouts capture medium-term trends. 1w HMA(21) filters for primary trend alignment (bull/bear). Volume > 1.5x 20-period average confirms institutional participation. ATR-based stoploss limits drawdown. Works in bull markets via trend continuation and bear markets via short breakdowns. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1908_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
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
    
    # Calculate 1w HMA(21)
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).rolling(window=half_period, min_periods=half_period).mean().values
        wma_full = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        hma = 2 * wma_half - wma_full
        hma = pd.Series(hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
        return hma
    
    hma_1w = calculate_hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_1w, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === HTF: 1d data for ATR stoploss ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 12h Indicators: Donchian(20) and Volume MA(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), ATR(14), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss: 2 * ATR against position
            stoploss_hit = False
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * entry_atr:
                    stoploss_hit = True
            else:  # Short
                if price > entry_price + 2.0 * entry_atr:
                    stoploss_hit = True
            
            # Exit on Donchian opposite breakout (trailing)
            donch_exit = False
            if position_side > 0:  # Long exit on lower band break
                if price < donch_low[i]:
                    donch_exit = True
            else:  # Short exit on upper band break
                if price > donch_high[i]:
                    donch_exit = True
            
            if stoploss_hit or donch_exit:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper band AND 1w uptrend
            if trend_1w_aligned[i] > 0 and price > donch_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_1d_aligned[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower band AND 1w downtrend
            elif trend_1w_aligned[i] < 0 and price < donch_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_1d_aligned[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals