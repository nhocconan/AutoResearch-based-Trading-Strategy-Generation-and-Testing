#!/usr/bin/env python3
"""
1h Donchian(20) Breakout + 4h Trend + Volume Spike
Hypothesis: 1h timeframe with Donchian breakouts aligned to 4h trend direction 
and volume confirmation captures momentum while avoiding chop. 4h trend provides 
structural bias (price above/below 4h EMA20 = bull/bear bias). Volume spike 
(>1.5x average) confirms participation. Works in bull/bear by following 4h trend.
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian20_4htrend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 4h EMA20 for trend bias
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 20:
        ema_4h[19] = np.mean(close_4h[:20])
        for i in range(20, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 18) / 20
    
    # Trend bias: above EMA20 = bullish, below = bearish
    trend_bias_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_bias_aligned = align_htf_to_ltf(prices, df_4h, trend_bias_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR trend flip
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.20
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR trend flip
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.20
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + 4h trend + volume
            # Minimum bars since exit: only allow new entry after 10 bars flat
            if bars_since_exit >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with bullish 4h trend and volume
                if bull_breakout and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: bearish breakout with bearish 4h trend and volume
                elif bear_breakout and trend_bias_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals