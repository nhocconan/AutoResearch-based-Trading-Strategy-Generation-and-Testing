#!/usr/bin/env python3
"""
Experiment #120: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 4h timeframe, filtered by 1d HMA(21) trend direction and confirmed by 4h volume spike (>1.8x 20-bar average), captures strong momentum moves while avoiding choppy markets. The strategy uses discrete position sizing (0.25) and ATR-based stoploss (2.5x ATR(14)) to manage risk. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag while maintaining statistical significance. Works in both bull and bear markets by following the 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = np.full_like(close_1d, np.nan)
        wma_full = np.full_like(close_1d, np.nan)
        
        if len(close_1d) >= half_len:
            wma_half[half_len-1:] = wma(close_1d, half_len)
        if len(close_1d) >= 21:
            wma_full[20:] = wma(close_1d, 21)
        
        raw_hma = 2 * wma_half - wma_full
        hma_21 = np.full_like(raw_hma, np.nan)
        if len(raw_hma) >= sqrt_len:
            hma_21[sqrt_len-1:] = wma(raw_hma, sqrt_len)
        
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high[19:] = high_series.rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = low_series.rolling(window=20, min_periods=20).min().values[19:]
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vol_series = pd.Series(volume)
        vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # ATR(14) for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for j in range(1, n):
            tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        atr[13:] = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values[13:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Use 1d HMA direction ---
        # Need previous close to determine HMA slope
        if i == 0:
            hma_rising = True  # Default for first bar
            hma_falling = False
        else:
            hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
            hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below Donchian low (trailing exit)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes above Donchian high (trailing exit)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume confirmation in uptrend
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above upper channel
            hma_rising and                   # 1d HMA trending up
            volume_spike                     # Volume confirmation
        )
        
        # Short: Price breaks below Donchian low with volume confirmation in downtrend
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below lower channel
            hma_falling and                  # 1d HMA trending down
            volume_spike                     # Volume confirmation
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