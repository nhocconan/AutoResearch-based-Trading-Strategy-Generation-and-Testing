#!/usr/bin/env python3
"""
Experiment #2994: 1h Strategy with 4h/1d HTF Filters
HYPOTHESIS: Use 4h Donchian(20) breakout direction + 1d EMA(50) trend filter + volume spike confirmation.
Only take longs when 4h price > Donchian high AND 1d close > EMA50 AND volume > 2.0x 20-bar MA.
Only take shorts when 4h price < Donchian low AND 1d close < EMA50 AND volume > 2.0x 20-bar MA.
1h timeframe used only for precise entry timing. Session filter (08-20 UTC) reduces noise.
Target: 60-150 total trades over 4 years = 15-37/year. Discrete position size 0.20 to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2994_1h_donchian20_4h_dir_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for Donchian channels (20-period) - Call ONCE before loop ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels on 4h
    lookback = 20
    highest_high_4h = pd.Series(high_4h).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=lookback, min_periods=lookback).min().values
    
    # === HTF: 1d data for EMA(50) trend filter - Call ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC (pre-compute hour array) ===
    # prices.index is already DatetimeIndex, .hour works directly
    hours = prices.index.hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size - discrete level to minimize churn
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: reverse on opposite signal ---
        if in_position:
            # Check for reverse signal
            if position_side == 1:  # Currently long
                # Reverse to short if 4h breaks below Donchian low AND 1d bearish AND volume spike
                if (i < len(lowest_low_4h) and not np.isnan(lowest_low_4h[i]) and
                    close[i] < lowest_low_4h[i] and
                    i < len(ema_1d) and not np.isnan(ema_1d[i]) and
                    close[i] < ema_1d[i] and
                    vol_ratio[i] > 2.0):
                    position_side = -1
                    signals[i] = -SIZE
                else:
                    signals[i] = SIZE  # maintain long
            else:  # Currently short
                # Reverse to long if 4h breaks above Donchian high AND 1d bullish AND volume spike
                if (i < len(highest_high_4h) and not np.isnan(highest_high_4h[i]) and
                    close[i] > highest_high_4h[i] and
                    i < len(ema_1d) and not np.isnan(ema_1d[i]) and
                    close[i] > ema_1d[i] and
                    vol_ratio[i] > 2.0):
                    position_side = 1
                    signals[i] = SIZE
                else:
                    signals[i] = -SIZE  # maintain short
            continue
        
        # --- New Position Entry Logic ---
        # Skip if any HTF data is NaN
        if (i >= len(highest_high_4h) or i >= len(lowest_low_4h) or
            i >= len(ema_1d) or
            np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or
            np.isnan(ema_1d[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Long entry: 4h price > Donchian high + 1d close > EMA50 + volume spike
        if (price > highest_high_4h[i] and
            price > ema_1d[i] and
            vol_ratio[i] > 2.0):
            position_side = 1
            signals[i] = SIZE
        
        # Short entry: 4h price < Donchian low + 1d close < EMA50 + volume spike
        elif (price < lowest_low_4h[i] and
              price < ema_1d[i] and
              vol_ratio[i] > 2.0):
            position_side = -1
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals