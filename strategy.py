#!/usr/bin/env python3
"""
Experiment #282: 12h Donchian(20) Breakout + 1d EMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Combining 12h Donchian channel breakouts with 1d EMA trend alignment and volume confirmation creates a robust trend-following strategy. The 12h timeframe minimizes fee drag while capturing medium-term trends. Donchian breakouts identify structural price levels, 1d EMA ensures trend alignment, and volume spikes confirm institutional participation. ATR-based stoploss manages risk. Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w data for trend strength
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
            dm_plus_1w[i] = move_up if move_up > move_down and move_up > 0 else 0
            dm_minus_1w[i] = move_down if move_down > move_up and move_down > 0 else 0
        
        # Smoothed TR, DM+
        tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        dm_plus_14 = pd.Series(dm_plus_1w).rolling(window=14, min_periods=14).sum().values
        dm_minus_14 = pd.Series(dm_minus_1w).rolling(window=14, min_periods=14).sum().values
        
        # DI+ and DI-
        di_plus = np.where(tr_14 > 0, dm_plus_14 / tr_14 * 100, 0)
        di_minus = np.where(tr_14 > 0, dm_minus_14 / tr_14 * 100, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) > 0, abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
        adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
        
        # Align to 12h timeframe
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    else:
        adx_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(20-1, n):
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume Spike Detection (2x 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    if n >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (ADX > 25) ---
        if adx_1w_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian low break (trailing stop)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian high break (trailing stop)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation required
        if not volume_spike[i]:
            signals[i] = 0.0
            continue
        
        # Long: Break above Donchian high with price above 1d EMA (bullish alignment)
        if close[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Break below Donchian low with price below 1d EMA (bearish alignment)
        elif close[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals