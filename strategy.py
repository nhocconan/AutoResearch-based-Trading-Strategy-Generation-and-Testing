#!/usr/bin/env python3
"""
Experiment #1984: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian channel breakouts capture institutional trend participation. 
Weekly HMA(21) filter ensures alignment with higher timeframe trend. 
Volume spike (>1.5x 20-day average) confirms institutional interest. 
ATR-based stoploss limits drawdown. Works in both bull (breakouts continue) and bear (failed breakouts reverse quickly) markets.
Target: 50-120 total trades over 4 years (12-30/year).
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
    
    # Calculate Weekly HMA(21): Hull Moving Average
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    def hma(values, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if len(values) < period:
            return np.full_like(values, np.nan)
        wma_half = wma(values, half_period)
        wma_full = wma(values, period)
        # Need to align arrays: wma_half starts at index half_period-1
        raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
        hma_values = wma(raw_hma, sqrt_period)
        # Pad with NaN to match original length
        result = np.full_like(values, np.nan)
        start_idx = period - half_period + sqrt_period - 1
        end_idx = start_idx + len(hma_values)
        if end_idx <= len(values) and start_idx >= 0:
            result[start_idx:end_idx] = hma_values
        return result
    
    hma_21_1w = hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20), ATR(14), Volume MA(20) ===
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First bar has no prior close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume MA(20) for spike detection
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
    stop_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), ATR(14), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2 * ATR below entry
                if price <= entry_price - 2.0 * atr[i]:
                    exit_signal = True
                # Take profit: Donchian low touch (mean reversion)
                elif price <= donchian_low[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2 * ATR above entry
                if price >= entry_price + 2.0 * atr[i]:
                    exit_signal = True
                # Take profit: Donchian high touch (mean reversion)
                elif price >= donchian_high[i]:
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
        # Require weekly trend alignment for bias filter
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND weekly trend up
            if trend_bias > 0 and price > donchian_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_price = entry_price - 2.0 * atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND weekly trend down
            elif trend_bias < 0 and price < donchian_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_price = entry_price + 2.0 * atr[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals