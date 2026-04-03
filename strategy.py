#!/usr/bin/env python3
"""
Experiment #064: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: Daily Donchian channel breakouts aligned with weekly HMA trend and volume spikes
capture strong momentum moves in both bull and bear markets. Weekly HMA filters for
trend direction, while daily breakouts provide precise entries. Volume confirmation
avoids false breakouts. Target: 30-100 trades over 4 years on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_064_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: HMA(21) for trend direction ===
    def calculate_hma(series, period):
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(s, p):
            if p <= 0:
                return np.full_like(s, np.nan)
            weights = np.arange(1, p + 1)
            return np.convolve(s, weights / weights.sum(), mode='valid')
        
        # Handle edge cases with padding
        if len(s) < period:
            return np.full_like(s, np.nan)
            
        wma_full = np.convolve(series, np.arange(1, period + 1) / (period * (period + 1) / 2), mode='valid')
        wma_half = np.convolve(series, np.arange(1, half_period + 1) / (half_period * (half_period + 1) / 2), mode='valid')
        
        # Need to align arrays properly
        hma_raw = 2 * wma_half - wma_full
        if len(hma_raw) < sqrt_period:
            return np.full_like(series, np.nan)
        hma = np.convolve(hma_raw, np.arange(1, sqrt_period + 1) / (sqrt_period * (sqrt_period + 1) / 2), mode='valid')
        
        # Pad to original length
        result = np.full_like(series, np.nan)
        start_idx = period - half_period  # Approximate offset
        end_idx = start_idx + len(hma)
        if end_idx <= len(series) and start_idx >= 0:
            result[start_idx:end_idx] = hma
        return result
    
    # Simpler approach: use EMA as proxy for HMA trend (more stable)
    # HMA(21) ~ EMA(21) for trend direction
    hma_21w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    hma_21w_aligned = align_htf_to_ltf(prices, df_1w, hma_21w)
    
    # === 1d Indicators: Donchian Channel (20) ===
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_price = close[i-1]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike threshold
        
        # --- Trend Filter: Weekly HMA ---
        # Price above weekly HMA = uptrend, below = downtrend
        is_uptrend = price > hma_21w_aligned[i]
        is_downtrend = price < hma_21w_aligned[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            if position_side > 0:  # Long position
                # Exit on Donchian lower break or trend reversal
                if price < donchian_lower[i] or (is_downtrend and bars_since_entry >= 3):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit on Donchian upper break or trend reversal
                if price > donchian_upper[i] or (is_uptrend and bars_since_entry >= 3):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 1 day
            if bars_since_entry < 1:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long entry: price breaks above Donchian upper + uptrend + volume
        if (price > donchian_upper[i-1] and 
            is_uptrend and 
            vol_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        
        # Short entry: price breaks below Donchian lower + downtrend + volume
        elif (price < donchian_lower[i-1] and 
              is_downtrend and 
              vol_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals