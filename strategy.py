#!/usr/bin/env python3
"""
Experiment #033: 4h Donchian(20) Breakout + 12h Volume Spike + 12h HMA Trend

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by 12h volume spike (>1.8x average) 
and 12h Hull Moving Average (HMA21) trend direction, creates a robust strategy that captures 
strong momentum moves in both bull and bear markets. The Donchian structure provides objective 
breakout levels, volume confirms institutional participation, and HMA filter ensures alignment 
with higher timeframe momentum. Targets 19-50 trades/year on 4h timeframe (75-200 total over 4 years) 
to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_volume_hma_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike and HMA trend (Call ONCE before loop) ===
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
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).mean().values
        wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_12h = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    peak_price = 0.0  # For trailing stop
    trough_price = 0.0  # For trailing stop
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (Trailing stop based on ATR) ---
        if in_position:
            # Calculate ATR(14) for dynamic stop
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update peak price
                if close[i] > peak_price:
                    peak_price = close[i]
                # Trailing stop: peak - 2.5 * ATR
                stop_level = peak_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price retouches Donchian middle (mean reversion signal)
                donchian_mid = (highest_high[i] + lowest_low[i]) / 2
                if close[i] < donchian_mid:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Update trough price
                if close[i] < trough_price:
                    trough_price = close[i]
                # Trailing stop: trough + 2.5 * ATR
                stop_level = trough_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price retouches Donchian middle
                donchian_mid = (highest_high[i] + lowest_low[i]) / 2
                if close[i] > donchian_mid:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: require significant spike
        volume_spike = vol_ratio_12h_aligned[i] > 1.8
        
        # Trend filter: HMA direction
        hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 else False
        
        # Long: Break above Donchian high with volume and rising HMA
        long_condition = (
            close[i] > highest_high[i] and 
            volume_spike and 
            hma_rising
        )
        
        # Short: Break below Donchian low with volume and falling HMA
        short_condition = (
            close[i] < lowest_low[i] and 
            volume_spike and 
            hma_falling
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            peak_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            trough_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals