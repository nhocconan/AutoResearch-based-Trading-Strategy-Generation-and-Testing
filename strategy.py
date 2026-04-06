#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d Close Trend + Volume Spike + ATR Stop
Hypothesis: Uses daily close relative to 200-day SMA for trend bias to avoid whipsaws in ranging markets.
Combines with Donchian breakouts for entry timing and volume confirmation for conviction.
Designed for low trade frequency (<25/year) to minimize fee drag and improve generalization.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1dtrend_vol_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # 1d close and 200-day SMA for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_200 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        sma_200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            sma_200[i] = (close_1d[i] + sma_200[i-1] * 199) / 200
    
    # Trend bias: close above SMA200 = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > sma_200, 1, -1)
    
    # Align to 4h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 200  # For SMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + 1d trend + volume spike
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with bullish 1d trend and volume
                if bull_breakout and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with bearish 1d trend and volume
                elif bear_breakout and trend_bias_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals