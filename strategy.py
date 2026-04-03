#!/usr/bin/env python3
"""
Experiment #201: 4h Donchian Breakout + HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe, filtered by 1d HMA trend direction and volume confirmation, capture institutional breakout moves in both bull and bear markets. The 1d HMA filter ensures alignment with higher timeframe trend, while volume spikes confirm genuine institutional participation. ATR-based stoploss manages risk. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_201_4h_donchian_hma_volume_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close for trend filter
    def calculate_hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if half_period < 1:
            half_period = 1
        if sqrt_period < 1:
            sqrt_period = 1
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean()
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw_hma = 2 * wma_half - wma_full
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
        return hma.values
    
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    trend_up_1d = close_1d > hma_21_1d
    trend_down_1d = close_1d < hma_21_1d
    
    # Align to 4h timeframe
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches opposite Donchian band or breaks with volume (continuation)
                if price <= lowest_low_20[i] and volume_spike:
                    # Breakdown with volume - reverse or exit
                    if trend_down_1d_aligned[i]:
                        # Continue short if 1d trend is down
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = -SIZE
                    else:
                        # Exit long
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                elif price >= highest_high_20[i] and not volume_spike:
                    # Take profit when breakout loses volume momentum
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches opposite Donchian band or breaks with volume (continuation)
                if price >= highest_high_20[i] and volume_spike:
                    # Breakout with volume - reverse or exit
                    if trend_up_1d_aligned[i]:
                        # Continue long if 1d trend is up
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = SIZE
                    else:
                        # Exit short
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                elif price <= lowest_low_20[i] and not volume_spike:
                    # Take profit when breakdown loses volume momentum
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Donchian breakout with volume spike and 1d trend alignment
        # Long: Price breaks above upper Donchian with volume in uptrend
        if (price > highest_high_20[i] and 
            trend_up_1d_aligned[i] and 
            volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below lower Donchian with volume in downtrend
        elif (price < lowest_low_20[i] and 
              trend_down_1d_aligned[i] and 
              volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals