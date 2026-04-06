#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d ADX Trend + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum aligned with 1d ADX trend (>25), volume confirms breakout strength. Using 12h timeframe with ADX trend filter targets 50-150 total trades over 4 years, suitable for bear/bull markets via directional filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1dadx_vol_v1"
timeframe = "12h"
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
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period ADX
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period + 1:
            return np.full(n, np.nan)
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        # Smoothed values
        atr_period = np.full(n, np.nan)
        plus_dm_period = np.full(n, np.nan)
        minus_dm_period = np.full(n, np.nan)
        if n >= period + 1:
            atr_period[period] = np.sum(tr[:period])
            plus_dm_period[period] = np.sum(plus_dm[:period])
            minus_dm_period[period] = np.sum(minus_dm[:period])
            for i in range(period + 1, n):
                atr_period[i] = (atr_period[i-1] * (period - 1) + tr[i-1]) / period
                plus_dm_period[i] = (plus_dm_period[i-1] * (period - 1) + plus_dm[i-1]) / period
                minus_dm_period[i] = (minus_dm_period[i-1] * (period - 1) + minus_dm[i-1]) / period
        # Avoid division by zero
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        valid = atr_period != 0
        plus_di[valid] = (plus_dm_period[valid] / atr_period[valid]) * 100
        minus_di[valid] = (minus_dm_period[valid] / atr_period[valid]) * 100
        di_sum = plus_di + minus_di
        dx_valid = (di_sum != 0) & ~np.isnan(di_sum)
        dx[dx_valid] = (np.abs(plus_di[dx_valid] - minus_di[dx_valid]) / di_sum[dx_valid]) * 100
        # Smoothed DX
        adx = np.full(n, np.nan)
        if n >= 2 * period:
            adx[2 * period - 1] = np.mean(dx[period:2 * period])
            for i in range(2 * period, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian and indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i]):
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
            # Minimum holding period: only allow new entry after 10 bars flat
            if bars_since_entry >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # ADX trend filter: only trade if ADX > 25 (trending market)
                # Long when bullish breakout, short when bearish breakout
                trend_filter = adx_1d_aligned[i] > 25
                
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
12h Donchian(20) Breakout + 1d ADX Trend + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum aligned with 1d ADX trend (>25), volume confirms breakout strength. Using 12h timeframe with ADX trend filter targets 50-150 total trades over 4 years, suitable for bear/bull markets via directional filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1dadx_vol_v1"
timeframe = "12h"
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
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period ADX
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period + 1:
            return np.full(n, np.nan)
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        # Smoothed values
        atr_period = np.full(n, np.nan)
        plus_dm_period = np.full(n, np.nan)
        minus_dm_period = np.full(n, np.nan)
        if n >= period + 1:
            atr_period[period] = np.sum(tr[:period])
            plus_dm_period[period] = np.sum(plus_dm[:period])
            minus_dm_period[period] = np.sum(minus_dm[:period])
            for i in range(period + 1, n):
                atr_period[i] = (atr_period[i-1] * (period - 1) + tr[i-1]) / period
                plus_dm_period[i] = (plus_dm_period[i-1] * (period - 1) + plus_dm[i-1]) / period
                minus_dm_period[i] = (minus_dm_period[i-1] * (period - 1) + minus_dm[i-1]) / period
        # Avoid division by zero
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        valid = atr_period != 0
        plus_di[valid] = (plus_dm_period[valid] / atr_period[valid]) * 100
        minus_di[valid] = (minus_dm_period[valid] / atr_period[valid]) * 100
        di_sum = plus_di + minus_di
        dx_valid = (di_sum != 0) & ~np.isnan(di_sum)
        dx[dx_valid] = (np.abs(plus_di[dx_valid] - minus_di[dx_valid]) / di_sum[dx_valid]) * 100
        # Smoothed DX
        adx = np.full(n, np.nan)
        if n >= 2 * period:
            adx[2 * period - 1] = np.mean(dx[period:2 * period])
            for i in range(2 * period, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian and indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i]):
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
            # Minimum holding period: only allow new entry after 10 bars flat
            if bars_since_entry >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # ADX trend filter: only trade if ADX > 25 (trending market)
                # Long when bullish breakout, short when bearish breakout
                trend_filter = adx_1d_aligned[i] > 25
                
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