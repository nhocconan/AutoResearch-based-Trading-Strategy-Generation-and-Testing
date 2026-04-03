#!/usr/bin/env python3
"""
Experiment #318: 1d Donchian Breakout + 1w HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on the 1d timeframe, confirmed by 1w HMA(21) trend alignment and volume spikes, capture strong momentum moves while filtering false breakouts. The weekly HMA ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws in ranging markets. Volume confirmation ensures institutional participation. Targets 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to minimize fee drag while capturing high-probability trend continuation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_volume_v1"
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
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_2x_sub = 2 * wma_half - wma_full
        hma_21 = wma(wma_2x_sub, sqrt_len)
        
        # Pad beginning with NaN
        hma_21_padded = np.full(len(close_1w), np.nan)
        hma_21_padded[half_len:] = hma_21
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume average (Call ONCE before loop) ===
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 1d Indicators ===
    # Donchian Channel (20)
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_h = high_series.rolling(window=20, min_periods=20).max().values
        donchian_l = low_series.rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_series = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean()
        atr = atr_series.values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(atr[i]) or np.isnan(hma_21_aligned[i]) or 
            np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1] if i > 0 else False
        hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1] if i > 0 else False
        
        # --- Volume Confirmation: Require volume spike (> 1.8x weekly average) ---
        volume_spike = vol_ratio_1w_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss and Donchian opposite) ---
        if in_position:
            if position_side > 0:  # Long position
                # Stoploss: 2.5 * ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                # Time exit: price re-enters Donchian channel
                if low[i] < stop_level or close[i] <= donchian_l[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Trailing stop: update highest since entry
                highest_since_entry = max(highest_since_entry, high[i])
                # Trail at 2.0 * ATR from high
                trail_level = highest_since_entry - 2.0 * atr[i]
                if low[i] < trail_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5 * ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                # Time exit: price re-enters Donchian channel
                if high[i] > stop_level or close[i] >= donchian_h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Trailing stop: update lowest since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Trail at 2.0 * ATR from low
                trail_level = lowest_since_entry + 2.0 * atr[i]
                if high[i] > trail_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian H + weekly HMA rising + volume spike
        long_condition = (
            close[i] > donchian_h[i] and 
            hma_rising and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian L + weekly HMA falling + volume spike
        short_condition = (
            close[i] < donchian_l[i] and 
            hma_falling and 
            volume_spike
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals