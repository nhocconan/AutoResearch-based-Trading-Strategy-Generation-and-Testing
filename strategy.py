#!/usr/bin/env python3
"""
Experiment #2899: 6h Elder Ray Power + ADX Regime Filter
HYPOTHESIS: Elder Ray (Bull/Bear Power) measures bull/bear strength relative to EMA13.
ADX > 25 filters for trending markets, < 20 for ranging. Long when Bull Power > 0 and ADX rising,
Short when Bear Power < 0 and ADX rising. Uses 6h timeframe for medium-term signals with
lower frequency than 1h/30m, reducing fee drag while capturing sustained moves in both bull
and bear regimes. Weekly pivot from 1d data adds directional bias: only take longs when
price > weekly pivot, shorts when price < weekly pivot. Volume confirmation (>1.5x average)
ensures breakout conviction. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2899_6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot bias (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from daily OHLC (last 5 days average)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    weekly_pivot = pd.Series(pivot_1d).rolling(window=5, min_periods=5).mean().values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Bull Power (High - EMA13) and Bear Power (Low - EMA13) ===
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: ADX(14) for trend strength ===
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    tr14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    adx[13:] = pd.Series(dx[13:]).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: Elder Ray power reverses OR ADX weakens (<20) OR price crosses EMA13
            if position_side > 0:  # Long
                if bull_power[i] <= 0 or adx[i] < 20 or price < ema13[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if bear_power[i] >= 0 or adx[i] < 20 or price > ema13[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Get weekly pivot bias
            price_vs_pivot = price - weekly_pivot_aligned[i]
            
            # Long entry: Bull Power positive, ADX > 25 (trending), price above weekly pivot
            if bull_power[i] > 0 and adx[i] > 25 and price_vs_pivot > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Bear Power negative, ADX > 25 (trending), price below weekly pivot
            elif bear_power[i] < 0 and adx[i] > 25 and price_vs_pivot < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals