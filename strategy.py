#!/usr/bin/env python3
"""
Experiment #1880: 4h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. Combined with 1d EMA filter for trend alignment, volume confirmation (>1.5x average), and ATR-based stoploss (2.5x), this strategy avoids false breakouts and choppy markets. Works in both bull and bear markets by following the 1d trend direction. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.25 to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1880_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend filter ===
    def hull_moving_average(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma
    
    hma_21 = hull_moving_average(close, 21)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), HMA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_21[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR stoploss or reverse signal ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: price < entry_price - 2.5 * ATR
                if price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
                # Reverse signal: price breaks below Donchian low AND HMA turns down
                elif price < donchian_low[i] and hma_21[i] < hma_21[i-1]:
                    exit_signal = True
                # Exit if 1d trend flips
                elif trend_1d_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Stoploss: price > entry_price + 2.5 * ATR
                if price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
                # Reverse signal: price breaks above Donchian high AND HMA turns up
                elif price > donchian_high[i] and hma_21[i] > hma_21[i-1]:
                    exit_signal = True
                # Exit if 1d trend flips
                elif trend_1d_aligned[i] > 0:
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
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Require volume confirmation (> 1.5x average)
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Long entry: price breaks above Donchian high AND price > HMA (uptrend)
            if trend_bias > 0 and price > donchian_high[i] and price > hma_21[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND price < HMA (downtrend)
            elif trend_bias < 0 and price < donchian_low[i] and price < hma_21[i]:
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