#!/usr/bin/env python3
"""
Experiment #941: 4h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian channel breakouts on 4h timeframe capture strong momentum moves. 
Filter by 1d HMA trend direction to avoid counter-trend trades. Require volume spike (>1.5x) 
for confirmation. Uses discrete position sizing (0.25) to limit drawdown. Target: 75-200 
total trades over 4 years (19-50/year) on 4h timeframe. Works in bull via breakouts, 
in bear via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_941_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) on 1d close
    def hull_moving_average(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma1 = pd.Series(arr).ewm(span=half_period, adjust=False).mean()
        wma2 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw_hma = 2 * wma1 - wma2
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
        return hma.values
    
    hma_1d = hull_moving_average(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 4h Indicators: Donchian channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
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
            
            # Optional: time-based exit after 16 bars (~2.6d on 4h) to avoid overtrading
            if bars_since_entry > 16:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine trend direction from 1d HMA
            # Uptrend: price > HMA, Downtrend: price < HMA
            # Use previous bar's HMA to avoid look-ahead
            hma_now = hma_1d_aligned[i]
            hma_prev = hma_1d_aligned[i-1] if i > 0 else hma_now
            
            # Long: price breaks above Donchian upper AND 1d HMA trending up
            if price > highest_high[i] and hma_now > hma_prev:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND 1d HMA trending down
            elif price < lowest_low[i] and hma_now < hma_prev:
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