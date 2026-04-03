#!/usr/bin/env python3
"""
Experiment #346: 4h Donchian(20) breakout + 1d HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by 1d HMA trend direction and 
confirmed by volume spikes, capture strong momentum moves while avoiding false breakouts in 
choppy markets. The 1d HMA ensures alignment with higher timeframe trend, reducing whipsaws. 
Volume confirmation ensures institutional participation. Targets 20-50 trades/year on 4h 
timeframe (80-200 total over 4 years) to minimize fee drag while maintaining statistical 
significance. Works in both bull (breakouts continuation) and bear (breakdowns continuation) 
markets by following the 1d trend filter.
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
        # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        wma_2xhalf = 2 * wma_half
        
        # Align lengths
        min_len = min(len(wma_2xhalf), len(wma_full))
        if min_len > 0 and len(wma_2xhalf) >= half_len and len(wma_full) >= 21:
            diff = wma_2xhalf[-min_len:] - wma_full[-min_len:]
            hma_raw = wma(diff, sqrt_len)
            # Pad to original length
            hma_21 = np.full(len(close_1d), np.nan)
            start_idx = len(close_1d) - len(hma_raw)
            if start_idx >= 0 and len(hma_raw) > 0:
                hma_21[start_idx:] = hma_raw
            else:
                hma_21 = np.full(len(close_1d), np.nan)
        else:
            hma_21 = np.full(len(close_1d), np.nan)
        
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel (20) - using previous 20 bars (exclude current)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1d HMA ---
        # HMA slope: current HMA vs previous HMA
        if i >= 1:
            hma_slope = hma_21_aligned[i] - hma_21_aligned[i-1]
            hma_uptrend = hma_slope > 0
            hma_downtrend = hma_slope < 0
        else:
            hma_uptrend = False
            hma_downtrend = False
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian High with volume and 1d HMA uptrend
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            hma_uptrend
        )
        
        # Short: Price breaks below Donchian Low with volume and 1d HMA downtrend
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            hma_downtrend
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