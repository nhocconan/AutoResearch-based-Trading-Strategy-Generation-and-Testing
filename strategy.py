#!/usr/bin/env python3
"""
Experiment #254: 1h Donchian(20) breakout + 4h/1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts on 1h aligned with 4h trend and 1d weekly pivot (R1/S1) direction capture high-probability moves. Volume confirmation (>1.8x average) filters weak breakouts. Uses 4h for trend filter and 1d for pivot levels, 1h only for entry timing to minimize overtrading. Target: 60-150 total trades over 4 years (15-37/year). Uses discrete sizing (0.20) to minimize fee drag. Includes session filter (08-20 UTC) to avoid low-liquidity hours.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_254_1h_donchian20_4h_1d_weekly_pivot_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # === HTF: 4h data for trend (EMA21) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for weekly pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # Prior week high
    week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)    # Prior week low
    week_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)  # Prior week close
    
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    
    # Align to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === 1h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr_1h = np.zeros(n)
    tr_1h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_1h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Enough for 20-period indicators and 5-day weekly pivot
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Trend Filter (4h EMA21): Only long in uptrend, short in downtrend ---
        uptrend_4h = price > ema_4h_aligned[i]
        downtrend_4h = price < ema_4h_aligned[i]
        
        # --- Pivot Direction Logic (1d weekly pivot) ---
        # Long bias: price above R1 (bullish)
        # Short bias: price below S1 (bearish)
        long_bias = price > r1_aligned[i]
        short_bias = price < s1_aligned[i]
        
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
                # Exit on breakout down with volume if bearish bias
                if breakout_down and volume_spike and short_bias:
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
                # Exit on breakout up with volume if bullish bias
                if breakout_up and volume_spike and long_bias:
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
        # Require volume spike + breakout conditions + trend + pivot bias alignment
        if volume_spike:
            # Long: breakout up AND uptrend (4h) AND bullish bias (above R1)
            if breakout_up and uptrend_4h and long_bias:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down AND downtrend (4h) AND bearish bias (below S1)
            elif breakout_down and downtrend_4h and short_bias:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals