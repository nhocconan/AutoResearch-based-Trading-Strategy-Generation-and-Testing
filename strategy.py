#!/usr/bin/env python3
"""
Experiment #024: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Daily Donchian channel breakouts (20-period) with weekly HMA(21) trend filter and 
volume spike confirmation creates a low-frequency, high-edge strategy for BTC/ETH/SOL. 
The 1d timeframe minimizes fee drag while capturing major trend moves. Weekly HMA ensures 
we only trade in the direction of the higher timeframe trend, avoiding counter-trend 
whipsaws. Volume confirmation filters out breakouts lacking institutional participation. 
ATR-based stoploss manages risk. Targets 7-25 trades/year (30-100 total over 4 years) 
to overcome fee drag in bear markets like 2025.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_hma_1w_volume_v1"
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
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full(len(arr), np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        if len(wma_half) >= half_len and len(wma_full) >= 21:
            raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
            hma_21 = wma(raw_hma, sqrt_len)
            # Pad to match original length
            hma_21_padded = np.full(len(close_1w), np.nan)
            hma_21_padded[half_len:half_len+len(hma_21)] = hma_21
            hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
        else:
            hma_21_aligned = np.full(n, np.nan)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_ratio = np.divide(volume, vol_ma_20, out=np.full(n, 1.0), where=vol_ma_20!=0)
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = max(100, lookback, 20)  # Ensure enough data
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(volume_ratio[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * entry_atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price re-enters Donchian channel (failed breakout)
                if close[i] <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * entry_atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price re-enters Donchian channel (failed breakdown)
                if close[i] >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: require > 1.8x average volume
        volume_confirm = volume_ratio[i] > 1.8
        
        # Trend filter: HMA direction
        hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
        hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        
        # Long: Donchian breakout above upper band with volume and uptrend
        long_condition = (
            close[i] > highest_high[i] and  # Breakout above Donchian high
            volume_confirm and
            hma_rising
        )
        
        # Short: Donchian breakdown below lower band with volume and downtrend
        short_condition = (
            close[i] < lowest_low[i] and  # Breakdown below Donchian low
            volume_confirm and
            hma_falling
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr_14[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr_14[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals