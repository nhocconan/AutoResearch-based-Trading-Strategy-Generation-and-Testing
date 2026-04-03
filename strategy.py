#!/usr/bin/env python3
"""
Experiment #300: 4h Donchian Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian(20) breakout on 4h timeframe, filtered by 1d HMA(21) trend and 
12h volume spike (>1.5x average), creates a robust trend-following strategy that 
works in both bull and bear markets. The Donchian structure provides clear breakout 
levels, HMA filters for higher timeframe trend direction, and volume confirms 
institutional participation. Targets 19-50 trades/year on 4h timeframe (75-200 total 
over 4 years) to minimize fee drag while capturing strong directional moves.
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
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        wma_half = np.array([wma(close_1d[max(0, i-half_len):i+1], half_len) 
                            if i >= half_len-1 else np.nan for i in range(len(close_1d))])
        wma_full = np.array([wma(close_1d[max(0, i-21):i+1], 21) 
                            if i >= 20 else np.nan for i in range(len(close_1d))])
        hma_21_raw = 2 * wma_half - wma_full
        hma_21 = np.array([wma(hma_21_raw[max(0, i-sqrt_len):i+1], sqrt_len) 
                          if i >= sqrt_len-1 else np.nan for i in range(len(hma_21_raw))])
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Calculate Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
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
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
        hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.5
        
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
                # Exit on Donchian low break (trailing stop)
                if close[i] < donchian_low[i]:
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
                # Exit on Donchian high break (trailing stop)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with HMA rising and volume spike
        long_condition = (
            close[i] > donchian_high[i] and 
            hma_rising and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian low with HMA falling and volume spike
        short_condition = (
            close[i] < donchian_low[i] and 
            hma_falling and 
            volume_spike
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