#!/usr/bin/env python3
"""
Experiment #946: 4h Donchian(20) Breakout + HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: 4h Donchian breakouts capture institutional order flow. Long when price breaks above Donchian(20) high with HMA(21) uptrend and volume spike (>1.8x avg). Short when price breaks below Donchian(20) low with HMA downtrend and volume spike. Uses ATR-based stoploss (2.0) and discrete position sizing (0.30) to limit drawdown. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_946_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime filter (optional, can be removed if not needed) ===
    # Keeping 1d HTF for potential future use but not currently used in signals
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: Donchian channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend filter ===
    def hma(arr, period):
        half = arr.copy()
        half[period//2:] = 2 * pd.Series(arr).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values[period//2:] - \
                         pd.Series(arr).ewm(span=period, min_periods=period, adjust=False).mean().values[period//2:]
        sqrt_period = int(np.sqrt(period))
        return pd.Series(half).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    hma_21 = hma(close, 21)
    hma_21_prev = np.roll(hma_21, 1)
    hma_21_prev[0] = hma_21[0]
    hma_uptrend = hma_21 > hma_21_prev
    hma_downtrend = hma_21 < hma_21_prev
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 21, 20)  # sufficient for Donchian, HMA, volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 28 bars (~7d on 4h) to avoid overtrading
            if bars_since_entry > 28:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Breakout continuation: price breaks above Donchian high with HMA uptrend
            if price > highest_high[i] and hma_uptrend[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Breakout continuation: price breaks below Donchian low with HMA downtrend
            elif price < lowest_low[i] and hma_downtrend[i]:
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