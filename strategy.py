#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA Trend + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum, weekly EMA from 1w timeframe filters for institutional bias, volume confirms breakout strength, ATR stoploss limits drawdown. Designed for low trade frequency (target 30-100 total over 4 years) to minimize fee decay. Works in bull/bear by only trading with higher timeframe EMA bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1wema_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA trend (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_1w[i] = close_1w[i] * 0.039216 + ema_1w[i-1] * 0.960784
    
    # Align weekly EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 50)  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine bias based on weekly EMA
        bullish_bias = close_1w[-1] > ema_1w[-1] if len(close_1w) > 0 else False  # Use latest weekly value for bias
        bearish_bias = close_1w[-1] < ema_1w[-1] if len(close_1w) > 0 else False
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry (using 14-period ATR approximated)
            # Simplified ATR: use 20-period high-low range as proxy
            if i >= 20:
                atr_proxy = np.mean(high[i-20:i] - low[i-20:i])
            else:
                atr_proxy = np.mean(high[:i+1] - low[:i+1])
            
            if close[i] < lowest_low or close[i] < entry_price - 2.0 * atr_proxy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if i >= 20:
                atr_proxy = np.mean(high[i-20:i] - low[i-20:i])
            else:
                atr_proxy = np.mean(high[:i+1] - low[:i+1])
            
            if close[i] > highest_high or close[i] > entry_price + 2.0 * atr_proxy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA bias
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Use weekly EMA bias from aligned array (current value)
            weekly_bullish = ema_1w_aligned[i] > 0 and len(close_1w) > 0 and close_1w[-1] > ema_1w[-1]
            weekly_bearish = ema_1w_aligned[i] > 0 and len(close_1w) > 0 and close_1w[-1] < ema_1w[-1]
            
            if i >= 20 and bull_breakout and volume_filter and weekly_bullish:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif i >= 20 and bear_breakout and volume_filter and weekly_bearish:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals