#!/usr/bin/env python3
"""
Experiment #381: 4h Donchian(20) breakout + 1d HMA trend + 1w volume spike + ATR stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by 1d HMA trend direction and 1w volume spike,
captures strong trending moves while avoiding false breakouts in choppy markets. Uses higher timeframes
(1d/1w) for signal validation to reduce whipsaw. Targets 20-50 trades/year (80-200 total over 4 years)
to minimize fee decay while maintaining statistical significance. Designed to work in both bull and bear
markets by requiring volume confirmation and trend alignment, preventing entries against the primary trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_hma_1w_vol_v1"
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
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        wma_half = pd.Series(close_1d).rolling(window=half_len, min_periods=half_len).apply(
            lambda x: np.dot(x, np.arange(1, half_len+1)) / np.arange(1, half_len+1).sum(), raw=True
        ).values
        wma_full = pd.Series(close_1d).rolling(window=21, min_periods=21).apply(
            lambda x: np.dot(x, np.arange(1, 22)) / np.arange(1, 22).sum(), raw=True
        ).values
        
        hma_raw = 2 * wma_half - wma_full
        hma_21 = pd.Series(hma_raw).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
            lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.arange(1, sqrt_len+1).sum(), raw=True
        ).values
        
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume spike (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 4h Indicators: Donchian(20) channels ===
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= donchian_period:
        # Calculate rolling max/min for Donchian channels
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
        donchian_low = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(100, donchian_period)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss and trailing stop) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update highest high since entry
                highest_since_entry = max(highest_since_entry, high[i])
                
                # Stoploss: 2.5 * ATR below entry OR 1.5 * ATR trailing from high
                stop_level = max(
                    entry_price - 2.5 * atr_14,
                    highest_since_entry - 1.5 * atr_14
                )
                
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                    
                # Exit on Donchian low break (contrarian exit)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            else:  # Short position
                # Update lowest low since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                
                # Stoploss: 2.5 * ATR above entry OR 1.5 * ATR trailing from low
                stop_level = min(
                    entry_price + 2.5 * atr_14,
                    lowest_since_entry + 1.5 * atr_14
                )
                
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                    
                # Exit on Donchian high break (contrarian exit)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filter: price above/below 1d HMA
        price_above_hma = close[i] > hma_21_aligned[i]
        price_below_hma = close[i] < hma_21_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio_1w_aligned[i] > 1.8
        
        # Long: Donchian high breakout with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above Donchian high
            volume_spike and                 # Volume confirmation
            price_above_hma                  # Trend alignment (uptrend)
        )
        
        # Short: Donchian low breakdown with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below Donchian low
            volume_spike and                 # Volume confirmation
            price_below_hma                  # Trend alignment (downtrend)
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals