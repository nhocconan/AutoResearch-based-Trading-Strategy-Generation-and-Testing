#!/usr/bin/env python3
"""
6h_WeeklyDonchian_Breakout_1dTrendRegime_v1
Hypothesis: On 6h timeframe, Donchian(20) breakout aligned with 1d EMA50 trend regime and volume confirmation (>1.5x 20-period average) captures strong directional moves. 
In bull regime (daily close > daily EMA50), favor longs on upper band breakout; in bear regime (daily close < daily EMA50), favor shorts on lower band breakout. 
Volume confirmation ensures institutional participation. Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for daily trend regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Donchian(20) breakout ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 2 days (8 * 6h = 48h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        daily_ema = ema_50_1d_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = price > daily_ema
        is_bear = price < daily_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above upper Donchian band
                long_condition = (price > highest_high[i]) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below lower Donchian band
                short_condition = (price < lowest_low[i]) and vol_conf
            
            if is_bull and long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif is_bear and short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR approximation using Donchian width)
            donchian_width = highest_high[i] - lowest_low[i]
            if donchian_width > 0:
                atr_approx = donchian_width / 2.0  # rough ATR approximation
                if position == 1:
                    if price < entry_price - 2.5 * atr_approx:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    # Time-based exit
                    elif bars_since_entry >= max_hold_bars:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.25
                else:  # position == -1
                    if price > entry_price + 2.5 * atr_approx:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    # Time-based exit
                    elif bars_since_entry >= max_hold_bars:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = -0.25
            else:
                # Fallback: time-based exit only
                if bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchian_Breakout_1dTrendRegime_v1"
timeframe = "6h"
leverage = 1.0