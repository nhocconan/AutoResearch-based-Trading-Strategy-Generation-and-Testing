#!/usr/bin/env python3
"""
Experiment #2994: 1h Strategy with 4h/1d HTF Direction Filter
HYPOTHESIS: Use 4h Donchian(20) breakouts + 1d EMA(50) trend filter for signal direction.
Only take entries in direction of higher timeframe trend. 1h timeframe used only for
precise entry timing via volume confirmation (>1.5x 20-period average). Session filter
(08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years = 15-37/year.
Position size fixed at 0.20 to manage drawdown. Stoploss at 2.5*ATR.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2994_1h_donchian20_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for Donchian channels (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels on 4h
    lookback = 20
    highest_high_4h = pd.Series(high_4h).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align 4h Donchian levels to 1h (automatically shift(1) for completed bars only)
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # === HTF: 1d data for EMA(50) trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Session filter: 08-20 UTC (pre-compute hours array) ===
    # prices.index is already DatetimeIndex with datetime64[ns]
    hours = prices.index.hour  # Vectorized, no conversion needed
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, lookback, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high_4h_aligned[i]) or np.isnan(lowest_low_4h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Stoploss: 2.5*ATR against position
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
        # Require volume confirmation (> 1.5x average)
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Get HTF signals
            price_vs_4h_high = price - highest_high_4h_aligned[i]
            price_vs_4h_low = price - lowest_low_4h_aligned[i]
            price_vs_ema_1d = price - ema_1d_aligned[i]
            
            # Long entry: price breaks above 4h Donchian high AND above 1d EMA (bullish alignment)
            if price_vs_4h_high > 0 and price_vs_ema_1d > 0:
                in_position = True
                position_side = 1
                entry_price = price
                signals[i] = SIZE
            # Short entry: price breaks below 4h Donchian low AND below 1d EMA (bearish alignment)
            elif price_vs_4h_low < 0 and price_vs_ema_1d < 0:
                in_position = True
                position_side = -1
                entry_price = price
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals