#!/usr/bin/env python3
"""
Experiment #3514: 1h Donchian Breakout + 4h EMA Trend + 1d Volume Spike Filter
HYPOTHESIS: 1h Donchian(20) breakouts aligned with 4h EMA(50) trend and confirmed by 1d volume spikes (>2x average) capture momentum with controlled frequency. 
4h EMA provides medium-term trend filter to avoid counter-trend whipsaws, while 1d volume spike ensures institutional participation. 
Position size 0.20. Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Uses 4h for trend direction and 1d for volume confirmation, 1h only for entry timing via Donchian breakout.
Works in bull (breakouts above rising EMA) and bear (breakouts below falling EMA) via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3514_1h_donchian20_4h_ema_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # auto shift(1) for completed bars
    
    # === HTF: 1d data for volume spike filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate volume MA(20) on 1d for spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones_like(volume_1d)
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)  # auto shift(1)
    
    # === 1h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_1h = 20
    highest_high_1h = pd.Series(high).rolling(window=lookback_1h, min_periods=lookback_1h).max().values
    lowest_low_1h = pd.Series(low).rolling(window=lookback_1h, min_periods=lookback_1h).min().values
    
    # === 1h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_1h, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_1h[i]) or np.isnan(lowest_low_1h[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        if volume_spike:
            # Determine trend bias from 4h EMA
            price_vs_ema = price - ema_4h_aligned[i]
            
            # Long entry: price breaks above 1h Donchian high with bullish trend (above EMA)
            if (price > highest_high_1h[i] and 
                price_vs_ema > 0):  # Above 4h EMA = bullish bias
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 1h Donchian low with bearish trend (below EMA)
            elif (price < lowest_low_1h[i] and 
                  price_vs_ema < 0):  # Below 4h EMA = bearish bias
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals