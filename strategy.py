#!/usr/bin/env python3
"""
Experiment #1860: 4h Donchian(20) breakout + 1d EMA(50) trend + volume confirmation
HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. Filtering by 1d EMA(50) ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws in ranging markets. Volume confirmation (>1.5x average) adds conviction to breakouts. This combination has proven effective across multiple symbols in the 4h timeframe with tight entry conditions (target: 75-200 trades over 4 years). Works in both bull and bear markets by following the 1d trend direction. Discrete position sizing of 0.30 minimizes fee churn while controlling drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1860_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss and re-entry prevention
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20) and EMA(50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or reverse signal ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss: 2.5 * ATR(14) approximation using 20-period range
            # Approximate ATR as average true range over 20 periods
            if i >= 20:
                tr_values = np.zeros(20)
                for j in range(20):
                    idx = i - j
                    if idx == 0:
                        tr_values[j] = high[idx] - low[idx]
                    else:
                        tr_values[j] = max(
                            high[idx] - low[idx],
                            abs(high[idx] - close[idx-1]),
                            abs(low[idx] - close[idx-1])
                        )
                atr_approx = np.mean(tr_values)
            else:
                atr_approx = (highest_20[i] - lowest_20[i]) * 0.2  # fallback
            
            stoploss_distance = 2.5 * atr_approx
            
            # Check stoploss
            if position_side > 0:  # Long position
                if price < entry_price - stoploss_distance:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakout in opposite direction
                elif price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price > entry_price + stoploss_distance:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakout in opposite direction
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position if no exit signal
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper band AND 1d trend is up
            if trend_bias > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower band AND 1d trend is down
            elif trend_bias < 0 and price < lowest_20[i]:
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