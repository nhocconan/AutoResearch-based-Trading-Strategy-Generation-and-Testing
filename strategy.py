#!/usr/bin/env python3
"""
Experiment #248: 12h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike

HYPOTHESIS: 12h Donchian channel breakouts filtered by weekly pivot direction (price > weekly pivot = bullish bias, 
price < weekly pivot = bearish bias) and volume spikes (>2.0x average) capture strong momentum moves with 
reduced false breakouts. Weekly pivot provides structural support/resistance from higher timeframe (1w). 12h timeframe 
targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while capturing significant moves. 
Works in both bull (breakouts with volume) and bear (failed breaks reverse sharply) markets. Uses ATR-based 
stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_248_12h_donchian_weekly_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot calculation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points from 1w data (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    weekly_pivot = np.full(n, np.nan)
    
    if len(df_1w) >= 2:  # Need at least 2 weeks of data
        # Align 1w data to LTF index for shifting
        # Create series indexed by 1w open_time
        df_1w_indexed = df_1w.set_index('open_time')
        
        # Calculate prior week's OHLC using shift(1) on the indexed series
        prior_week_high = df_1w_indexed['high'].shift(1).values
        prior_week_low = df_1w_indexed['low'].shift(1).values
        prior_week_close = df_1w_indexed['close'].shift(1).values
        
        # Calculate weekly pivot for each prior week
        prior_week_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
        
        # Create series aligned with 1w index
        weekly_pivot_series = pd.Series(index=df_1w_indexed.index, data=prior_week_pivot)
        
        # Align to LTF (12h) timeframe with shift(1) for completed bars only
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_series.values)
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
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF weekly pivot, ATR, and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Filter: Price > pivot = bullish bias, Price < pivot = bearish bias ---
        price_above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
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
        # Long: Donchian breakout up + volume spike + price above weekly pivot
        long_condition = breakout_up and volume_spike and price_above_weekly_pivot
        
        # Short: Donchian breakout down + volume spike + price below weekly pivot
        short_condition = breakout_down and volume_spike and price_below_weekly_pivot
        
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

</think>