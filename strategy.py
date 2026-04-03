#!/usr/bin/env python3
"""
Experiment #365: 12h Donchian(20) breakout + 1d HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, filtered by 1d HMA(21) trend direction 
and confirmed by 12h volume spike (>2x average), capture strong momentum moves in both bull 
and bear markets. The 1d HMA filter ensures alignment with higher timeframe trend, reducing 
whipsaw during ranging periods. Volume confirmation ensures institutional participation. 
Target: 12-37 trades/year (50-150 total over 4 years) on 12h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_vol_v1"
timeframe = "12h"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        wma_2x_sub = 2 * wma_half - wma_full[
            len(wma_full) - len(wma_2x_sub):] if len(wma_2x_sub) < len(wma_full) else wma_full[:len(wma_2x_sub)]
        ]
        # Pad to original length
        wma_2x_sub_padded = np.full(len(close_1d), np.nan)
        wma_2x_sub_padded[half_len:half_len+len(wma_2x_sub)] = wma_2x_sub
        hma_21 = wma(wma_2x_sub_padded[~np.isnan(wma_2x_sub_padded)], sqrt_len)
        hma_21_padded = np.full(len(close_1d), np.nan)
        start_idx = len(close_1d) - len(hma_21)
        hma_21_padded[start_idx:] = hma_21
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_padded)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 12h data for volume confirmation (Call ONCE before loop) ===
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
    
    # === 12h Indicators ===
    # Calculate Donchian channels (20-period) on 12h
    donchian_period = 20
    upper_12h = np.full(len(df_12h), np.nan)
    lower_12h = np.full(len(df_12h), np.nan)
    
    if len(df_12h) >= donchian_period:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        for i in range(donchian_period - 1, len(df_12h)):
            upper_12h[i] = np.max(high_12h[i-donchian_period+1:i+1])
            lower_12h[i] = np.min(low_12h[i-donchian_period+1:i+1])
    
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
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
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        hma_rising = hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1] if i > 0 else False
        hma_falling = hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1] if i > 0 else False
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
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
                # Exit if price crosses below Donchian lower (trailing exit)
                if close[i] < lower_12h_aligned[i]:
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
                # Exit if price crosses above Donchian upper (trailing exit)
                if close[i] > upper_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper with volume confirmation and HMA rising
        long_condition = (
            close[i] > upper_12h_aligned[i] and 
            volume_spike and 
            hma_rising
        )
        
        # Short: Price breaks below Donchian lower with volume confirmation and HMA falling
        short_condition = (
            close[i] < lower_12h_aligned[i] and 
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