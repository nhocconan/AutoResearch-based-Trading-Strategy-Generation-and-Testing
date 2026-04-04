#!/usr/bin/env python3
"""
Experiment #4097: 4h Donchian(20) breakout + 1d EMA(50) + 1w EMA(200) trend filter + volume confirmation
HYPOTHESIS: Donchian breakouts on 4h aligned with both 1d EMA(50) (medium-term) and 1w EMA(200) (long-term) trend direction and volume confirmation capture strong continuation moves while minimizing trades. Dual timeframe trend filter (1d+1w) provides robust regime adaptation for both bull and bear markets by only allowing breakouts when aligned with intermediate AND long-term trends. Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4097_4h_donchian20_1d_ema50_1w_ema200_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA(50) for intermediate trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w EMA(200) for long-term trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    else:
        ema_200_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(20) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10)  # DC lookback, vol MA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Dual HTF trend bias: 
            price_above_1d_ema = price > ema_50_1d_aligned[i]
            price_below_1d_ema = price < ema_50_1d_aligned[i]
            price_above_1w_ema = price > ema_200_1w_aligned[i]
            price_below_1w_ema = price < ema_200_1w_aligned[i]
            
            # Breakout logic: 
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long conditions: above BOTH 1d EMA(50) AND 1w EMA(200) + upper Donchian breakout
            long_entry = breakout_up and price_above_1d_ema and price_above_1w_ema
            
            # Short conditions: below BOTH 1d EMA(50) AND 1w EMA(200) + lower Donchian breakout
            short_entry = breakout_down and price_below_1d_ema and price_below_1w_ema
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
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