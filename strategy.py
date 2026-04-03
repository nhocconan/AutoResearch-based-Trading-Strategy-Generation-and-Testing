#!/usr/bin/env python3
"""
Experiment #1934: 1h Donchian Breakout + 4h Trend + 1d Volume Filter
HYPOTHESIS: Use 4h EMA(50) for trend direction, 1d volume spike for conviction, and 1h Donchian(20) breakout for entry timing. 
This combines multi-timeframe alignment (4h trend + 1h entry) with volume confirmation to avoid false breakouts. 
Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1934_1h_donchian_4h_trend_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(50) for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_4h = np.where(close_4h > ema_50_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for volume filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones_like(volume_1d)
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_filter_1d = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1h Indicators: Donchian Channel(20) ===
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(50, donchian_period)  # sufficient for EMA(50) and Donchian
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(vol_filter_1d[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: trailing stop at 2.5 * ATR(14) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for dynamic stoploss
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate ATR using Wilder's smoothing
            if bars_since_entry == 1:
                atr = tr
            else:
                atr = (atr * 13 + tr) / 14
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price drops below entry - 2.5 * ATR
                if price < entry_price - 2.5 * atr:
                    exit_signal = True
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price < highest_high[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price rises above entry + 2.5 * ATR
                if price > entry_price + 2.5 * atr:
                    exit_signal = True
                # Exit if price re-enters Donchian channel
                elif price > lowest_low[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 4h trend alignment
        trend_bias = trend_4h_aligned[i]
        # Require 1d volume spike (> 1.8x average)
        volume_spike = vol_filter_1d[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 4h trend up
            if trend_bias > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 4h trend down
            elif trend_bias < 0 and price < lowest_low[i]:
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