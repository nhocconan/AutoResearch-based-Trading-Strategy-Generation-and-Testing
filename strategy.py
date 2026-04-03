#!/usr/bin/env python3
"""
Experiment #309: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by 1d HMA trend direction and 
1d volume confirmation, captures strong momentum moves while avoiding false breakouts in choppy 
markets. The Donchian structure provides objective breakout levels, HMA filter ensures alignment 
with higher timeframe trend, and volume confirmation requires institutional participation. 
Targets 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag.
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
            return np.convolve(values, weights/weights.sum(), mode='valid')
        
        wma_half = np.full_like(close_1d, np.nan)
        wma_full = np.full_like(close_1d, np.nan)
        
        if len(close_1d) >= half_len:
            wma_half[half_len-1:] = wma(close_1d, half_len)
        if len(close_1d) >= 21:
            wma_full[20:] = wma(close_1d, 21)
        
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = np.full_like(close_1d, np.nan)
        if len(raw_hma) >= sqrt_len:
            hma_21_1d[sqrt_len-1:] = wma(raw_hma[sqrt_len-1:], sqrt_len)
        
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Donchian Channel (20)
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(20, n):
            donchian_h[i] = np.max(high[i-20:i])
            donchian_l[i] = np.min(low[i-20:i])
    
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
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        hma_rising = hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1] if i > 0 else False
        hma_falling = hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1] if i > 0 else False
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
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
            
            # Exit on opposite Donchian break
            if position_side > 0 and close[i] < donchian_l[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and close[i] > donchian_h[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian H + HMA rising + volume spike
        long_condition = (
            close[i] > donchian_h[i] and 
            hma_rising and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian L + HMA falling + volume spike
        short_condition = (
            close[i] < donchian_l[i] and 
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