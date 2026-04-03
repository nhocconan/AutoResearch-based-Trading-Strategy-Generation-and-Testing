#!/usr/bin/env python3
"""
Experiment #348: 12h Donchian(20) Breakout + 1w HMA Trend + Volume Spike

HYPOTHESIS: Donchian channel breakouts on 12h timeframe combined with 1-week HMA trend filter
and volume spike confirmation creates a robust strategy that works in both bull and bear markets.
The 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
HMA on 1w provides smooth trend direction, while volume confirms institutional participation.
ATR-based stoploss manages risk. Discrete position sizing (0.25) reduces churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_2x_sub = 2 * wma_half - wma_full
        hma_21 = wma(wma_2x_sub, sqrt_len)
        
        # Pad to match original length
        hma_21_padded = np.full(len(close_1w), np.nan)
        hma_21_padded[half_len:] = hma_21
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Donchian(20) on 12h (using lookback of 20 bars) ---
        if i >= 20:
            lookback_high = np.max(high[i-20:i])
            lookback_low = np.min(low[i-20:i])
        else:
            # Not enough lookback - use available data
            lookback_high = np.max(high[:i]) if i > 0 else high[i]
            lookback_low = np.min(low[:i]) if i > 0 else low[i]
        
        # --- Trend Filter: HMA direction on 1w ---
        # Need previous HMA value to determine slope
        if i >= warmup + 1:
            hma_now = hma_21_aligned[i]
            hma_prev = hma_21_aligned[i-1]
            hma_rising = hma_now > hma_prev
            hma_falling = hma_now < hma_prev
        else:
            hma_rising = False
            hma_falling = False
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian break (mean reversion tendency)
                if close[i] < lookback_low:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian break
                if close[i] > lookback_high:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian HIGH with volume in uptrend
        long_condition = (
            close[i] > lookback_high and 
            volume_spike and 
            hma_rising
        )
        
        # Short: Price breaks below Donchian LOW with volume in downtrend
        short_condition = (
            close[i] < lookback_low and 
            volume_spike and 
            hma_falling
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals