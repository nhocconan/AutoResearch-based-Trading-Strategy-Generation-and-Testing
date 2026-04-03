#!/usr/bin/env python3
"""
Experiment #060: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by 1d HMA trend direction and 
12h volume confirmation, capture strong momentum moves while avoiding false breakouts in ranging markets. 
The strategy uses discrete position sizing (0.25) to limit fee impact, with ATR-based stoploss 
for risk management. Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag 
while maintaining statistical significance. Works in both bull (breakout longs) and bear (breakout shorts) 
markets by following the higher timeframe trend.
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
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA(21) = WMA(2*WMA(n/2) - WMA(n)) 
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).rolling(window=half, min_periods=half).mean().values
        wma_full = pd.Series(close_1d).rolling(window=21, min_periods=21).mean().values
        wma_2x_half = 2 * wma_half
        wma_diff = wma_2x_half - wma_full
        hma_21 = pd.Series(wma_diff).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 12h data for volume confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Donchian Channel (20)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i >= 20:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
        else:
            donchian_high[i] = high[i]
            donchian_low[i] = low[i]
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate dynamic stoploss
            if position_side > 0:  # Long
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian touch (mean reversion)
                if close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian touch
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Determine trend direction from 1d HMA
        if i > 0:
            hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
            hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        else:
            hma_rising = hma_falling = False
        
        # Volume confirmation: require > 1.3x average volume
        volume_ok = vol_ratio_12h_aligned[i] > 1.3
        
        # Long: Donchian breakout above upper band in uptrend with volume
        long_breakout = (close[i] > donchian_high[i]) and hma_rising and volume_ok
        
        # Short: Donchian breakdown below lower band in downtrend with volume
        short_breakout = (close[i] < donchian_low[i]) and hma_falling and volume_ok
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals