#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h ADX Trend + Volume Filter
Hypothesis: Donchian breakouts on 6h capture momentum aligned with 12h ADX trend (ADX>25), volume confirms breakout strength. Targets 50-150 total trades over 4 years with strict entry criteria. Works in both bull and bear markets by filtering trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12hadx_vol_v1"
timeframe = "6h"
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
    
    # Load 12h data once before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX on 12h data
    adx_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 30:
        # True Range
        tr_12h = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        # Directional Movement
        plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                           np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
        minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                            np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
        
        # Smoothing (Wilder's smoothing)
        tr_14 = np.full(len(tr_12h), np.nan)
        plus_dm_14 = np.full(len(plus_dm), np.nan)
        minus_dm_14 = np.full(len(minus_dm), np.nan)
        
        if len(tr_12h) >= 14:
            tr_14[13] = np.sum(tr_12h[:14])
            plus_dm_14[13] = np.sum(plus_dm[:14])
            minus_dm_14[13] = np.sum(minus_dm[:14])
            for i in range(14, len(tr_12h)):
                tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr_12h[i]
                plus_dm_14[i] = plus_dm_14[i-1] - (plus_dm_14[i-1] / 14) + plus_dm[i]
                minus_dm_14[i] = minus_dm_14[i-1] - (minus_dm_14[i-1] / 14) + minus_dm[i]
        
        # Directional Indicators
        plus_di = np.full(len(close_12h), np.nan)
        minus_di = np.full(len(close_12h), np.nan)
        if len(tr_14) >= 14:
            for i in range(14, len(tr_14)):
                if not np.isnan(tr_14[i]) and tr_14[i] != 0:
                    plus_di[i] = (plus_dm_14[i] / tr_14[i]) * 100
                    minus_di[i] = (minus_dm_14[i] / tr_14[i]) * 100
        
        # ADX
        dx = np.full(len(close_12h), np.nan)
        if len(plus_di) >= 14:
            for i in range(27, len(close_12h)):  # Need 14+13 for DX smoothing
                if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                    di_sum = plus_di[i] + minus_di[i]
                    if di_sum != 0:
                        dx[i] = np.abs(plus_di[i] - minus_di[i]) / di_sum * 100
            
            # Smooth DX to get ADX
            adx_12h = np.full(len(close_12h), np.nan)
            if len(dx) >= 28:
                adx_12h[27] = np.nanmean(dx[14:28])  # First ADX after 27 periods
                for i in range(28, len(dx)):
                    if not np.isnan(dx[i]):
                        adx_12h[i] = (adx_12h[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(adx_12h_aligned[i]):
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
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Trend filter: only trade if ADX > 25 (trending market)
                trend_filter = adx_12h_aligned[i] > 25
                
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

#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h ADX Trend + Volume Filter
Hypothesis: Donchian breakouts on 6h capture momentum aligned with 12h ADX trend (ADX>25), volume confirms breakout strength. Targets 50-150 total trades over 4 years with strict entry criteria. Works in both bull and bear markets by filtering trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12hadx_vol_v1"
timeframe = "6h"
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
    
    # Load 12h data once before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX on 12h data
    adx_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 30:
        # True Range
        tr_12h = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        # Directional Movement
        plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                           np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
        minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                            np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
        
        # Smoothing (Wilder's smoothing)
        tr_14 = np.full(len(tr_12h), np.nan)
        plus_dm_14 = np.full(len(plus_dm), np.nan)
        minus_dm_14 = np.full(len(minus_dm), np.nan)
        
        if len(tr_12h) >= 14:
            tr_14[13] = np.sum(tr_12h[:14])
            plus_dm_14[13] = np.sum(plus_dm[:14])
            minus_dm_14[13] = np.sum(minus_dm[:14])
            for i in range(14, len(tr_12h)):
                tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr_12h[i]
                plus_dm_14[i] = plus_dm_14[i-1] - (plus_dm_14[i-1] / 14) + plus_dm[i]
                minus_dm_14[i] = minus_dm_14[i-1] - (minus_dm_14[i-1] / 14) + minus_dm[i]
        
        # Directional Indicators
        plus_di = np.full(len(close_12h), np.nan)
        minus_di = np.full(len(close_12h), np.nan)
        if len(tr_14) >= 14:
            for i in range(14, len(tr_14)):
                if not np.isnan(tr_14[i]) and tr_14[i] != 0:
                    plus_di[i] = (plus_dm_14[i] / tr_14[i]) * 100
                    minus_di[i] = (minus_dm_14[i] / tr_14[i]) * 100
        
        # ADX
        dx = np.full(len(close_12h), np.nan)
        if len(plus_di) >= 14:
            for i in range(27, len(close_12h)):  # Need 14+13 for DX smoothing
                if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                    di_sum = plus_di[i] + minus_di[i]
                    if di_sum != 0:
                        dx[i] = np.abs(plus_di[i] - minus_di[i]) / di_sum * 100
            
            # Smooth DX to get ADX
            adx_12h = np.full(len(close_12h), np.nan)
            if len(dx) >= 28:
                adx_12h[27] = np.nanmean(dx[14:28])  # First ADX after 27 periods
                for i in range(28, len(dx)):
                    if not np.isnan(dx[i]):
                        adx_12h[i] = (adx_12h[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(adx_12h_aligned[i]):
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
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Trend filter: only trade if ADX > 25 (trending market)
                trend_filter = adx_12h_aligned[i] > 25
                
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