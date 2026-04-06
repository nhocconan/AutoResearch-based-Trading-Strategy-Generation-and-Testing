#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + Volume Spike + Weekly RSI Filter + ATR Stoploss
Hypothesis: 12h timeframe reduces trade frequency to optimal levels while capturing major trends.
Donchian breakouts with volume spike (>1.5x average) and weekly RSI filter (avoid extremes) 
work in both bull/bear markets by focusing on momentum with volatility confirmation.
ATR-based stoploss limits drawdown. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_volume_weeklyrsi_v1"
timeframe = "12h"
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
    
    # Weekly RSI for trend filter (using mtf_data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    rsi_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        # Calculate RSI for weekly
        delta = np.diff(close_1w)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(close_1w), np.nan)
        avg_loss = np.full(len(close_1w), np.nan)
        
        if len(gain) >= 14:
            avg_gain[13] = np.mean(gain[:14])
            avg_loss[13] = np.mean(loss[:14])
            
            for i in range(14, len(close_1w)):
                avg_gain[i] = (gain[i-1] * 13 + avg_gain[i-1]) / 14
                avg_loss[i] = (loss[i-1] * 13 + avg_loss[i-1]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1w = 100 - (100 / (1 + rs))
        rsi_1w[:13] = np.nan
    
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(rsi_1w_aligned[i]):
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
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Weekly RSI filter (avoid overbought/oversold extremes)
        rsi_filter = (rsi_1w_aligned[i] > 30) & (rsi_1w_aligned[i] < 70)
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + RSI filter
            # Minimum holding period: only allow new entry after 10 bars flat
            if bars_since_entry >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                if bull_breakout and volume_filter and rsi_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and rsi_filter:
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