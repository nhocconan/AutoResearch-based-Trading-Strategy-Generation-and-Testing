#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + ADX Trend Filter + ATR Stoploss
Hypothesis: Donchian breakouts with volume spike (>2x average) and strong trend (ADX>25) capture high-probability moves. ADX filter prevents whipsaws in ranging markets. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_vol_adx_v1"
timeframe = "4h"
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
    
    # 14-period ADX
    adx = np.full(n, np.nan)
    if n >= 14:
        # +DM and -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range (same as ATR calculation)
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        
        # Smoothed values
        tr14 = np.full(n, np.nan)
        plus_dm14 = np.full(n, np.nan)
        minus_dm14 = np.full(n, np.nan)
        
        if len(tr) >= 14:
            tr14[14] = np.sum(tr[:14])
            plus_dm14[14] = np.sum(plus_dm[:14])
            minus_dm14[14] = np.sum(minus_dm[:14])
            
            for i in range(15, n):
                tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i-1]
                plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / 14) + plus_dm[i-1]
                minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / 14) + minus_dm[i-1]
        
        # Directional Indicators
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        valid = ~np.isnan(tr14) & (tr14 != 0)
        if np.any(valid):
            plus_di[valid] = 100 * plus_dm14[valid] / tr14[valid]
            minus_di[valid] = 100 * minus_dm14[valid] / tr14[valid]
            dx[valid] = 100 * np.abs(plus_di[valid] - minus_di[valid]) / (plus_di[valid] + minus_di[valid])
        
        # ADX (smoothed DX)
        adx_smooth = np.full(n, np.nan)
        valid_dx = ~np.isnan(dx)
        if np.sum(valid_dx) >= 14:
            # First ADX value is average of first 14 DX
            first_idx = np.where(valid_dx)[0][13] if np.sum(valid_dx) >= 14 else -1
            if first_idx != -1:
                adx_smooth[first_idx] = np.mean(dx[valid_dx][:14])
                for i in range(first_idx + 1, n):
                    if valid_dx[i]:
                        adx_smooth[i] = (adx_smooth[i-1] * 13 + dx[i]) / 14
                adx = adx_smooth
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian and ADX
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(adx[i]):
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
        volume_filter = volume[i] > vol_ma * 2.0  # Increased threshold for fewer trades
        
        # ADX trend filter (strong trend)
        trend_filter = adx[i] > 25
        
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
            # Look for entries: Donchian breakout + volume + ADX trend filter
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                if bull_breakout and volume_filter and trend_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and trend_filter:
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

</think>