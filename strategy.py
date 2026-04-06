#!/usr/bin/env python3
"""
4h Donchian20 + 1d EMA Trend + Volume Spike
Hypothesis: Combines Donchian channel breakouts (20-period) with 1d EMA trend filter and volume spike confirmation.
In bull markets: Buy when price breaks above upper Donchian band with 1d uptrend and volume spike.
In bear markets: Sell when price breaks below lower Donchian band with 1d downtrend and volume spike.
Volume spike ensures institutional participation, reducing false breakouts.
Uses ATR-based stoploss (2x ATR) to manage risk.
Target: 75-200 trades over 4 years (19-50/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1dtrend_vol_v1"
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
    
    # 20-period ATR for stoploss and position sizing
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 19 + atr[i-1]) / 20
    
    # 20-period Donchian channels (highest high, lowest low over 20 periods)
    # Using rolling window with min_periods
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_channel[i] = np.max(high[i-20:i])
        lower_channel[i] = np.min(low[i-20:i])
    
    # 1d EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish (1), below = bearish (-1)
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend bias to 4h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Volume spike filter: current volume > 2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = max(20, 50)  # Need Donchian and EMA periods
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(trend_bias_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below lower Donchian band OR against 1d trend OR stoploss
            if (close[i] < lower_channel[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian band OR against 1d trend OR stoploss
            if (close[i] > upper_channel[i] or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries with minimum 8 bars between trades
            if bars_since_exit >= 8:
                # Long: breakout above upper Donchian with 1d uptrend and volume spike
                long_entry = (close[i] > upper_channel[i] and 
                             trend_bias_aligned[i] == 1 and 
                             volume_spike[i])
                
                # Short: breakout below lower Donchian with 1d downtrend and volume spike
                short_entry = (close[i] < lower_channel[i] and 
                              trend_bias_aligned[i] == -1 and 
                              volume_spike[i])
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                elif short_entry:
                    signals[i] = -0.25
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