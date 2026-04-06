#!/usr/bin/env python3
"""
6h Donchian(20) breakout with 1d ADX filter and volume confirmation
Hypothesis: Donchian breakouts capture momentum, filtered by 1d ADX>25 to ensure trending markets and volume confirmation for conviction. Works in both bull and bear by taking breakouts in direction of 1d trend. Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_adx_vol_v1"
timeframe = "6h"
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
    
    # Get 1d data for ADX and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Smoothed values
        tr_sum = np.zeros(n)
        dm_plus_sum = np.zeros(n)
        dm_minus_sum = np.zeros(n)
        
        # Initial values
        if len(tr) >= period:
            tr_sum[period] = np.sum(tr[:period])
            dm_plus_sum[period] = np.sum(dm_plus[:period])
            dm_minus_sum[period] = np.sum(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, n):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i-1]
            dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / period) + dm_plus[i-1]
            dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / period) + dm_minus[i-1]
        
        # Directional Indicators
        di_plus = np.full(n, np.nan)
        di_minus = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if tr_sum[i] > 0:
                di_plus[i] = 100 * dm_plus_sum[i] / tr_sum[i]
                di_minus[i] = 100 * dm_minus_sum[i] / tr_sum[i]
                if di_plus[i] + di_minus[i] > 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX (smoothed DX)
        adx = np.full(n, np.nan)
        if len(dx) >= 2 * period:
            # First ADX value
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            # Subsequent values
            for i in range(2*period, n):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # 1d trend: using EMA21 for direction
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 21:
        ema_1d[20] = np.mean(close_1d[:21])
        for i in range(21, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 19) / 21
    
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Get 1d data for volume
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) from 6h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # ADX filter: trending market (ADX > 25)
        adx_filter = adx_1d_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against 1d trend OR ADX weak
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_1d_aligned[i] == -1 or
                adx_1d_aligned[i] < 20 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against 1d trend OR ADX weak
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_1d_aligned[i] == 1 or
                adx_1d_aligned[i] < 20 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: upper/lower with 1d trend and filters
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish 1d trend + ADX + volume
                if bull_breakout and trend_1d_aligned[i] == 1 and adx_filter and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish 1d trend + ADX + volume
                elif bear_breakout and trend_1d_aligned[i] == -1 and adx_filter and volume_filter:
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
6h Donchian(20) breakout with 1d ADX filter and volume confirmation
Hypothesis: Donchian breakouts capture momentum, filtered by 1d ADX>25 to ensure trending markets and volume confirmation for conviction. Works in both bull and bear by taking breakouts in direction of 1d trend. Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_adx_vol_v1"
timeframe = "6h"
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
    
    # Get 1d data for ADX and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Smoothed values
        tr_sum = np.zeros(n)
        dm_plus_sum = np.zeros(n)
        dm_minus_sum = np.zeros(n)
        
        # Initial values
        if len(tr) >= period:
            tr_sum[period] = np.sum(tr[:period])
            dm_plus_sum[period] = np.sum(dm_plus[:period])
            dm_minus_sum[period] = np.sum(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, n):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i-1]
            dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / period) + dm_plus[i-1]
            dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / period) + dm_minus[i-1]
        
        # Directional Indicators
        di_plus = np.full(n, np.nan)
        di_minus = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if tr_sum[i] > 0:
                di_plus[i] = 100 * dm_plus_sum[i] / tr_sum[i]
                di_minus[i] = 100 * dm_minus_sum[i] / tr_sum[i]
                if di_plus[i] + di_minus[i] > 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX (smoothed DX)
        adx = np.full(n, np.nan)
        if len(dx) >= 2 * period:
            # First ADX value
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            # Subsequent values
            for i in range(2*period, n):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # 1d trend: using EMA21 for direction
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 21:
        ema_1d[20] = np.mean(close_1d[:21])
        for i in range(21, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 19) / 21
    
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Get 1d data for volume
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) from 6h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # ADX filter: trending market (ADX > 25)
        adx_filter = adx_1d_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against 1d trend OR ADX weak
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_1d_aligned[i] == -1 or
                adx_1d_aligned[i] < 20 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against 1d trend OR ADX weak
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_1d_aligned[i] == 1 or
                adx_1d_aligned[i] < 20 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: upper/lower with 1d trend and filters
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish 1d trend + ADX + volume
                if bull_breakout and trend_1d_aligned[i] == 1 and adx_filter and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish 1d trend + ADX + volume
                elif bear_breakout and trend_1d_aligned[i] == -1 and adx_filter and volume_filter:
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