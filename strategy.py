#!/usr/bin/env python3
"""
Experiment #2474: 1h Donchian(20) breakout with 4h/1d trend + volume + session filter
HYPOTHESIS: 1h timeframe is noisy but can work when using 4h/1d for trend direction and 1h only for precise entry timing. 
Donchian breakouts with volume confirmation and multi-timeframe trend alignment capture institutional participation. 
Session filter (08-20 UTC) reduces noise from low-liquidity periods. Discrete sizing (0.20) limits fee drag. 
Target: 75-150 total trades over 4 years (19-37/year) to balance opportunity with cost efficiency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2474_1h_donchian20_4h_1d_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for Donchian trend ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) channels
    highest_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    trend_4h = np.where(highest_20_4h > lowest_20_4h, 
                        np.where(close > highest_20_4h, 1, 
                                 np.where(close < lowest_20_4h, -1, 0)), 0)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d EMA for higher timeframe trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian(20) channels, Volume MA(20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any indicator invalid
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2*ATR below highest since entry (using Donchian width)
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below 1h Donchian low
                elif price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2*ATR above lowest since entry
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above 1h Donchian high
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry ---
        # Require both 4h and 1d trend alignment for stronger bias
        if trend_4h_aligned[i] == 0 or trend_1d_aligned[i] == 0:
            signals[i] = 0.0
            continue
            
        # Volume confirmation: spike > 1.5x average
        volume_spike = vol_ratio[i] > 1.5
        
        if not volume_spike:
            signals[i] = 0.0
            continue
            
        # Long: price breaks above 1h Donchian high with uptrend on both 4h and 1d
        if (trend_4h_aligned[i] > 0 and trend_1d_aligned[i] > 0 and 
            price > highest_20[i]):
            in_position = True
            position_side = 1
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        # Short: price breaks below 1h Donchian low with downtrend on both 4h and 1d
        elif (trend_4h_aligned[i] < 0 and trend_1d_aligned[i] < 0 and 
              price < lowest_20[i]):
            in_position = True
            position_side = -1
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals