#!/usr/bin/env python3
"""
Experiment #3559: 6h Donchian Breakout + 12h Trend Filter + Volume Spike
HYPOTHESIS: 6h Donchian(20) breakouts with 12h EMA(50) trend filter and volume confirmation capture medium-term momentum. 
12h EMA(50) provides smoother trend direction than 6h alone, reducing whipsaw. Volume confirms breakout strength. 
Position size 0.25. Target: 80-180 total trades over 4 years (20-45/year).
Uses 12h for trend filter, 6h only for entry timing and risk management.
Works in bull (continuation above EMA50) and bear (continuation below EMA50) via price channels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3559_6h_donchian20_12h_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA50 trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 6h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_6h = 20
    highest_high_6h = pd.Series(high).rolling(window=lookback_6h, min_periods=lookback_6h).max().values
    lowest_low_6h = pd.Series(low).rolling(window=lookback_6h, min_periods=lookback_6h).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_6h, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_6h[i]) or np.isnan(lowest_low_6h[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                # Exit if price breaks below 6h Donchian low (trend reversal)
                elif price < lowest_low_6h[i]:
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
                # Exit if price breaks above 6h Donchian high (trend reversal)
                elif price > highest_high_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Determine trend bias from 12h EMA50
            price_vs_ema = price - ema_50_12h_aligned[i]
            
            # Long entry: price breaks above 6h Donchian high with bullish bias (above EMA50)
            if (price > highest_high_6h[i] and 
                price_vs_ema > 0):  # Above EMA50 = bullish bias
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 6h Donchian low with bearish bias (below EMA50)
            elif (price < lowest_low_6h[i] and 
                  price_vs_ema < 0):  # Below EMA50 = bearish bias
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