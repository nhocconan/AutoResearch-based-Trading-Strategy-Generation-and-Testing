#!/usr/bin/env python3
"""
Experiment #2963: 4h Donchian Breakout + 12h HMA Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h capture medium-term trends with controlled trade frequency.
12-period HMA on 12h provides directional bias: only long when 12h HMA rising, short when falling.
Volume spike (>2.0x 20-period average) confirms breakout strength. ATR-based stoploss (2.5x ATR)
manages risk. This combination filters false breakouts while capturing strong trends in both bull
and bear markets. 4h timeframe targets 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2963_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(12) on 12h: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1, dtype=np.float64)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    n_12h = len(close_12h)
    half = 12 // 2
    sqrt_n = int(np.sqrt(12))
    
    wma_half = wma(close_12h, half)
    wma_full = wma(close_12h, 12)
    wma_2xhalf_minus_full = 2 * wma_half[-len(wma_full):] - wma_full
    hma_12h_raw = wma(wma_2xhalf_minus_full, sqrt_n)
    
    # Pad to match original length
    hma_12h = np.full(n_12h, np.nan)
    hma_12h[-(len(hma_12h_raw)):] = hma_12h_raw
    
    # Align to 4h timeframe (shifted by 1 for completed bars only)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate HMA slope for trend direction
    hma_slope = np.diff(hma_12h_aligned, prepend=np.nan)
    
    # === 4h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, lookback, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_slope[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if price moves 2.5*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Get 12h HMA trend direction
            trend_up = hma_slope[i] > 0
            trend_down = hma_slope[i] < 0
            
            # Long entry: price breaks above Donchian high with rising 12h HMA
            if price > highest_high[i] and trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with falling 12h HMA
            elif price < lowest_low[i] and trend_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals