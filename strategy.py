#!/usr/bin/env python3
"""
Experiment #257: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Combining Donchian channel breakouts with HMA trend alignment and volume confirmation on the 4h timeframe creates a robust trend-following strategy that works in both bull and bear markets. The Donchian(20) provides clear breakout signals, the HMA(21) filters for trend direction, and volume confirmation ensures institutional participation. ATR-based stoploss manages risk. Targets 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag while capturing strong trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_1d_v1"
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
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = np.full(len(close_1d), np.nan)
        wma_full = np.full(len(close_1d), np.nan)
        
        if len(close_1d) >= half_len:
            wma_half[half_len-1:] = wma(close_1d, half_len)
        if len(close_1d) >= 21:
            wma_full[20:] = wma(close_1d, 21)
        
        hma_input = 2 * wma_half - wma_full
        hma_21_1d = np.full(len(close_1d), np.nan)
        if len(hma_input) >= sqrt_len:
            hma_21_1d[sqrt_len-1:] = wma(hma_input, sqrt_len)
        
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w data for regime filter
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr_1w = np.zeros(len(close_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        
        # Directional Movement
        dm_plus_1w = np.zeros(len(close_1w))
        dm_minus_1w = np.zeros(len(close_1w))
        for i in range(1, len(close_1w)):
            move_up = high_1w[i] - high_1w[i-1]
            move_down = low_1w[i-1] - low_1w[i]
            if move_up > move_down and move_up > 0:
                dm_plus_1w[i] = move_up
            else:
                dm_plus_1w[i] = 0
            if move_down > move_up and move_down > 0:
                dm_minus_1w[i] = move_down
            else:
                dm_minus_1w[i] = 0
        
        # Smoothed TR, DM+
        atr_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = np.zeros(len(close_1w))
        di_minus = np.zeros(len(close_1w))
        valid_atr = atr_1w > 0
        di_plus[valid_atr] = 100 * dm_plus_smooth[valid_atr] / atr_1w[valid_atr]
        di_minus[valid_atr] = 100 * dm_minus_smooth[valid_atr] / atr_1w[valid_atr]
        
        # DX and ADX
        dx = np.zeros(len(close_1w))
        dx_denom = di_plus + di_minus
        valid_dx = dx_denom > 0
        dx[valid_dx] = 100 * np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / dx_denom[valid_dx]
        adx_1w = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align to 4h timeframe
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    else:
        adx_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: volume > 1.5 * average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ADX > 25 (trending market) ---
        if adx_1w_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
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
                # Take profit at 3R
                if close[i] >= entry_price + 3.0 * (entry_price - stop_level):
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
                # Take profit at 3R
                if close[i] <= entry_price - 3.0 * (stop_level - entry_price):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian High + price above HMA + volume confirmation
        if (close[i] > donchian_high[i] and 
            close[i] > hma_21_1d_aligned[i] and 
            volume_confirm[i]):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Price breaks below Donchian Low + price below HMA + volume confirmation
        elif (close[i] < donchian_low[i] and 
              close[i] < hma_21_1d_aligned[i] and 
              volume_confirm[i]):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals