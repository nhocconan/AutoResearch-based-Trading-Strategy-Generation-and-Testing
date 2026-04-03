#!/usr/bin/env python3
"""
Experiment #086: 4h Donchian(20) breakout + 1d HMA trend + 1w volume spike

HYPOTHESIS: Donchian(20) breakouts on 4h, filtered by 1d HMA(21) trend and 1w volume confirmation (>1.8x average),
capture medium-term momentum with controlled trade frequency. Volume spike confirms institutional participation.
Target: 20-50 trades/year on 4h (80-200 total over 4 years) to minimize fee drag. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_hma_vol_v2"
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
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).rolling(window=half_len, min_periods=half_len).mean().values
        wma_full = pd.Series(close_1d).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.ones(len(vol_1w))  # Start with neutral
        if len(vol_1w) > 20:
            vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Donchian(20) channels
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level to reduce churn)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if HTF data not ready
        if np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter
        price_above_hma = close[i] > hma_21_1d_aligned[i]
        price_below_hma = close[i] < hma_21_1d_aligned[i]
        
        # Volume confirmation: 1w volume > 1.8x 20-period average
        volume_spike = vol_ratio_1w_aligned[i] > 1.8
        
        # === Exit Logic ===
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:  # Stoploss hit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                if close[i] <= donchian_low[i]:  # Take profit at Donchian low
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:  # Stoploss hit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                if close[i] >= donchian_high[i]:  # Take profit at Donchian high
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # === Entry Logic ===
        # Long: Break above Donchian high + uptrend + volume spike
        if (close[i] > donchian_high[i] and 
            price_above_hma and 
            volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        
        # Short: Break below Donchian low + downtrend + volume spike
        elif (close[i] < donchian_low[i] and 
              price_below_hma and 
              volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        
        # Else remain flat
        else:
            signals[i] = 0.0
    
    return signals