#!/usr/bin/env python3
"""
Experiment #408: 12h Donchian(20) breakout + 1w HMA trend + 1d volume spike

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, filtered by 1w HMA trend direction and confirmed by 1d volume spike, captures medium-term trends while minimizing false breakouts. The 1w HMA provides robust trend filtering (works in both bull/bear markets), 1d volume spike ensures institutional participation, and Donchian(20) breakouts capture momentum. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian20_1w_hma_1d_vol_v1"
timeframe = "12h"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 12h Indicators ===
    # Calculate Donchian channel (20-period) on 12h
    # Need to get 12h OHLC data
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        # Donchian upper/lower = rolling max(high,20)/min(low,20)
        donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
        donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    else:
        donchian_upper_aligned = np.full(n, np.nan)
        donchian_lower_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction (rising/falling) ---
        # Calculate HMA slope using previous value
        if i > warmup:
            hma_slope = hma_21_aligned[i] - hma_21_aligned[i-1]
            hma_rising = hma_slope > 0
            hma_falling = hma_slope < 0
        else:
            hma_rising = True  # Neutral for warmup
            hma_falling = True
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using available data up to i
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
                # Exit if price breaks below Donchian lower (trend reversal)
                if close[i] < donchian_lower_aligned[i]:
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
                # Exit if price breaks above Donchian upper (trend reversal)
                if close[i] > donchian_upper_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper + HMA rising + volume spike
        long_condition = (
            close[i] > donchian_upper_aligned[i] and 
            hma_rising and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian lower + HMA falling + volume spike
        short_condition = (
            close[i] < donchian_lower_aligned[i] and 
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