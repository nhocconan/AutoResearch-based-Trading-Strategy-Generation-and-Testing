#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h RSI Filter + Volume Confirmation + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum, 12h RSI filters for overbought/oversold conditions to avoid counter-trend entries, volume confirms breakout strength, ATR stoploss limits drawdown. Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee decay. Works in bull/bear by only trading when RSI is not extreme.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12hrsi_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for RSI calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 14-period RSI on 12h
    rsi_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 15:
        delta = np.diff(close_12h)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        
        if len(gain) >= 14:
            avg_gain[13] = np.mean(gain[:14])
            avg_loss[13] = np.mean(loss[:14])
            
            for i in range(14, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
            rsi_12h[14:] = 100 - (100 / (1 + rs))
    
    # RSI thresholds: avoid extreme overbought (>70) and oversold (<30)
    rsi_not_overbought = rsi_12h < 70
    rsi_not_oversold = rsi_12h > 30
    
    # Align RSI and conditions to 6h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    rsi_not_overbought_aligned = align_htf_to_ltf(prices, df_12h, rsi_not_overbought)
    rsi_not_oversold_aligned = align_htf_to_ltf(prices, df_12h, rsi_not_oversold)
    
    # 14-period ATR on 12h for stoploss
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    atr_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 15:
        tr = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        atr_12h[0] = np.nan
        if len(tr) > 0:
            atr_12h[1] = tr[0]
            for i in range(2, len(atr_12h)):
                atr_12h[i] = (tr[i-1] * 13 + atr_12h[i-1]) / 14
    
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 15)  # For Donchian and RSI
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(rsi_not_overbought_aligned[i]) or 
            np.isnan(rsi_not_oversold_aligned[i]) or
            np.isnan(atr_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
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
            # Exit: price closes below Donchian lower OR RSI becomes overbought
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or 
                not rsi_not_overbought_aligned[i] or  # RSI >= 70
                close[i] < entry_price - 2.0 * atr_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR RSI becomes oversold
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or 
                not rsi_not_oversold_aligned[i] or  # RSI <= 30
                close[i] > entry_price + 2.0 * atr_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + RSI filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and volume_filter and rsi_not_overbought_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif i >= 20 and bear_breakout and volume_filter and rsi_not_oversold_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals