#!/usr/bin/env python3
"""
Experiment #072: 12h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike + Chop Filter

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, filtered by weekly pivot direction 
(price > weekly pivot = bullish bias, price < weekly pivot = bearish bias), volume spikes (>2.0x average), 
and choppiness regime (CHOP > 61.8 = ranging, avoid breakouts in chop) capture strong momentum 
moves with reduced false breakouts. Weekly pivot provides structural support/resistance on higher timeframe. 
12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while 
capturing significant moves. Chop filter avoids whipsaws in ranging markets. ATR-based stoploss 
manages risk. Works in both bull (breakouts with volume) and bear (failed breaks reverse sharply).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_072_12h_donchian_weekly_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 1d data (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    weekly_pivot = np.full(n, np.nan)
    
    if len(df_1d) >= 5:  # Need at least 5 days (1 week) of data
        # Create series indexed by 1d open_time
        df_1d_indexed = df_1d.set_index('open_time')
        
        # Resample to weekly using prior week's OHLC (shift(1) to avoid look-ahead)
        weekly_ohlc = df_1d_indexed.resample('W').agg({
            'high': 'max',
            'low': 'min', 
            'close': 'last'
        }).shift(1)  # Use prior week only
        
        # Calculate weekly pivot for each prior week
        weekly_pivot_values = (weekly_ohlc['high'] + weekly_ohlc['low'] + weekly_ohlc['close']) / 3.0
        
        # Create series aligned with weekly index
        weekly_pivot_series = pd.Series(index=weekly_ohlc.index, data=weekly_pivot_values.values)
        
        # Align to LTF (12h) timeframe with shift(1) for completed bars only
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_series.values)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 12h Indicators: Choppiness Index (14) for regime filter ===
    atr_14_chop = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_chop).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.full(n, np.nan)
    # Avoid division by zero
    denominator = np.log10(14) * (highest_high_14 - lowest_low_14)
    chop[13:] = 100 * np.log10(sum_atr_14[13:] / np.where(denominator[13:] != 0, denominator[13:], 1))
    chop[:13] = 50.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital) - balanced for 12h timeframe
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF weekly pivot, ATR, and chop
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Filter: Price > pivot = bullish bias, Price < pivot = bearish bias ---
        price_above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Chop Filter: Avoid breakouts in ranging markets (CHOP > 61.8) ---
        chop_filter = chop[i] <= 61.8  # Only allow breakouts when not excessively choppy
        
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
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above weekly pivot + chop filter
        long_condition = breakout_up and volume_spike and price_above_weekly_pivot and chop_filter
        
        # Short: Donchian breakout down + volume spike + price below weekly pivot + chop filter
        short_condition = breakout_down and volume_spike and price_below_weekly_pivot and chop_filter
        
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