#!/usr/bin/env python3
"""
Experiment #2209: 4h Donchian(20) breakout + 1d/1w HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: 4h Donchian channel breakouts with dual timeframe trend filtering (1d/1w HMA) capture swing momentum.
- Primary: 4h Donchian(20) breakout with volume > 1.5x 20-bar average (moderate threshold)
- HTF: 1d HMA(21) AND 1w HMA(21) trend filter (both must align for entry)
- Exit: ATR(14) trailing stop (2*ATR) or opposite Donchian channel touch
- Target: 75-200 total trades over 4 years (19-50/year) - proven range for 4h strategies
- Designed to work in bull markets (trend following) and bear markets (avoiding counter-trend trades)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2209_4h_donchian20_1d_1w_hma_vol_v1"
timeframe = "4h"
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
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21): Hull Moving Average
    def calculate_hma(arr, period):
        half_len = period // 2
        sqrt_len = int(np.sqrt(period))
        
        # WMA function
        def wma(data, window):
            if len(data) < window:
                return np.full_like(data, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights[::-1], mode='valid') / weights.sum()
        
        # Calculate WMAs
        wma_full = np.full(len(arr), np.nan)
        wma_half = np.full(len(arr), np.nan)
        
        for i in range(period-1, len(arr)):
            wma_full[i] = np.mean(arr[i-period+1:i+1] * np.arange(1, period+1))
        for i in range(half_len-1, len(arr)):
            wma_half[i] = np.mean(arr[i-half_len+1:i+1] * np.arange(1, half_len+1))
        
        # HMA = WMA(2*WMA_half - WMA_full, sqrt_len)
        wma_diff = 2 * wma_half - wma_full
        hma = np.full(len(arr), np.nan)
        for i in range(sqrt_len-1, len(arr)):
            if i >= half_len-1 and not np.isnan(wma_diff[i]):
                hma[i] = np.mean(wma_diff[i-sqrt_len+1:i+1] * np.arange(1, sqrt_len+1))
        return hma
    
    # Calculate 1d and 1w HMA
    hma_1d = calculate_hma(close_1d, 21)
    hma_1w = calculate_hma(close_1w, 21)
    
    # Trend: 1 if close > HMA, -1 otherwise
    trend_1d = np.where(close_1d > hma_1d, 1, -1)
    trend_1w = np.where(close_1w > hma_1w, 1, -1)
    
    # Align HTF trends to LTF
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 4h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for confirmation (moderate threshold to balance trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - conservative for risk management
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (mean reversion)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (mean reversion)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require BOTH 1d AND 1w trend alignment for strong bias filter
        trend_bias_1d = trend_1d_aligned[i]
        trend_bias_1w = trend_1w_aligned[i]
        
        # Only trade when both timeframes agree on trend direction
        trend_aligned = (trend_bias_1d > 0 and trend_bias_1w > 0) or (trend_bias_1d < 0 and trend_bias_1w < 0)
        
        # Volume confirmation: require volume spike (> 1.5x average - moderate threshold)
        volume_spike = vol_ratio[i] > 1.5
        
        if trend_aligned and volume_spike:
            # Long entry: price breaks above upper Donchian AND both trends up
            if trend_bias_1d > 0 and trend_bias_1w > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND both trends down
            elif trend_bias_1d < 0 and trend_bias_1w < 0 and price < donchian_lower[i]:
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