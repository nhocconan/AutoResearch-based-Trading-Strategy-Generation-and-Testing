#!/usr/bin/env python3
"""
Experiment #317: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: 4h Donchian channel breakouts filtered by 1d Hull Moving Average trend 
and volume spikes (>2.0x average) capture strong momentum moves with reduced false 
breakouts. The 1d HMA provides a longer-term trend filter (more stable than 12h), 
balancing responsiveness and smoothness. 4h timeframe targets 19-50 trades/year (75-200 total 
over 4 years) to minimize fee drag while capturing significant moves. Works in both 
bull (breakouts with volume) and bear (failed breaks reverse sharply) markets. Uses 
ATR-based stoploss for risk management.

OPTIMIZATIONS FROM PREVIOUS EXPERIMENTS:
- Increased volume threshold to 2.5x (from 2.0x) to reduce trade frequency to target range
- Added minimum holding period of 3 bars to reduce churn
- Adjusted warmup to 100 for better stability
- Maintained discrete position sizing at 0.25
- Added choppiness index regime filter to avoid whipsaws in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_317_4h_donchian_1d_hma_volume_chop_v1"
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
    
    # Calculate HMA(21) on 1d data
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).rolling(window=half_period, min_periods=half_period).mean().values
        wma_full = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        raw_hma = 2.0 * wma_half - wma_full
        hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
        return hma
    
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === HTF: 1d data for Choppiness Index regime filter ===
    def calculate_choppiness(high_arr, low_arr, close_arr, period):
        if len(close_arr) < period:
            return np.full_like(close_arr, np.nan)
        atr = np.zeros(len(close_arr))
        atr[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(close_arr)):
            atr[i] = max(high_arr[i] - low_arr[i], 
                         abs(high_arr[i] - close_arr[i-1]), 
                         abs(low_arr[i] - close_arr[i-1]))
        sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        ratio = sum_atr / (highest_high - lowest_low)
        chop = 100 * np.log10(ratio) / np.log10(period)
        return chop
    
    chop_1d = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Increased warmup for better stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d HMA Trend Filter: Price > HMA = bullish bias, Price < HMA = bearish bias ---
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.5x average) ---
        volume_spike = vol_ratio[i] > 2.5
        
        # --- Choppiness Regime Filter: Avoid ranging markets (CHOP > 61.8) ---
        # Only trade when market is trending (CHOP < 61.8)
        not_choppy = chop_1d_aligned[i] < 61.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
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
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above 1d HMA + not choppy
        long_condition = breakout_up and volume_spike and price_above_hma and not_choppy
        
        # Short: Donchian breakout down + volume spike + price below 1d HMA + not choppy
        short_condition = breakout_down and volume_spike and price_below_hma and not_choppy
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals