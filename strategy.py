#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d ADX Trend + Volume Spike + ATR Stop
Hypothesis: Combines price channel breakouts with 1d ADX trend filter for strong trends
and volume confirmation to capture momentum while avoiding chop.
Works in bull (breakouts with strong trend) and bear (strong downtrend breakdowns).
Designed for low trade frequency (~15-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1dadx_vol_v1"
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
    
    # 1d ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    adx_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        # True Range
        tr_1d = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        
        # Smooth TR, DM+ and DM-
        tr_sum = np.full(len(tr_1d), np.nan)
        dm_plus_sum = np.full(len(tr_1d), np.nan)
        dm_minus_sum = np.full(len(tr_1d), np.nan)
        
        if len(tr_1d) >= 14:
            tr_sum[13] = np.sum(tr_1d[:14])
            dm_plus_sum[13] = np.sum(dm_plus[:14])
            dm_minus_sum[13] = np.sum(dm_minus[:14])
            
            for i in range(14, len(tr_1d)):
                tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / 14) + tr_1d[i]
                dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / 14) + dm_plus[i]
                dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / 14) + dm_minus[i]
            
            # DI+ and DI-
            di_plus = np.full(len(tr_1d), np.nan)
            di_minus = np.full(len(tr_1d), np.nan)
            for i in range(14, len(tr_1d)):
                if tr_sum[i] != 0:
                    di_plus[i] = 100 * dm_plus_sum[i] / tr_sum[i]
                    di_minus[i] = 100 * dm_minus_sum[i] / tr_sum[i]
            
            # DX and ADX
            dx = np.full(len(tr_1d), np.nan)
            for i in range(14, len(tr_1d)):
                if di_plus[i] + di_minus[i] != 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
            
            adx_1d = np.full(len(tr_1d), np.nan)
            if len(dx) >= 28:  # Need 14 for DX smoothing + 14 for ADX
                adx_1d[27] = np.nanmean(dx[14:28])  # First ADX value
                for i in range(28, len(dx)):
                    adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Trend filter: ADX > 25 indicates strong trend
    trend_filter_1d = adx_1d > 25
    
    # Align to 4h timeframe
    trend_filter_aligned = align_htf_to_ltf(prices, df_1d, trend_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_filter_aligned[i]):
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
            # Exit: price closes below Donchian lower OR weak trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                not trend_filter_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR weak trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                not trend_filter_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + strong trend + volume spike
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with strong trend and volume
                if bull_breakout and trend_filter_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with strong trend and volume
                elif bear_breakout and trend_filter_aligned[i] and volume_filter:
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
4h Donchian(20) Breakout + 1d ADX Trend + Volume Spike + ATR Stop
Hypothesis: Combines price channel breakouts with 1d ADX trend filter for strong trends
and volume confirmation to capture momentum while avoiding chop.
Works in bull (breakouts with strong trend) and bear (strong downtrend breakdowns).
Designed for low trade frequency (~15-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1dadx_vol_v1"
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
    
    # 1d ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    adx_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        # True Range
        tr_1d = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        
        # Smooth TR, DM+ and DM-
        tr_sum = np.full(len(tr_1d), np.nan)
        dm_plus_sum = np.full(len(tr_1d), np.nan)
        dm_minus_sum = np.full(len(tr_1d), np.nan)
        
        if len(tr_1d) >= 14:
            tr_sum[13] = np.sum(tr_1d[:14])
            dm_plus_sum[13] = np.sum(dm_plus[:14])
            dm_minus_sum[13] = np.sum(dm_minus[:14])
            
            for i in range(14, len(tr_1d)):
                tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / 14) + tr_1d[i]
                dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / 14) + dm_plus[i]
                dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / 14) + dm_minus[i]
            
            # DI+ and DI-
            di_plus = np.full(len(tr_1d), np.nan)
            di_minus = np.full(len(tr_1d), np.nan)
            for i in range(14, len(tr_1d)):
                if tr_sum[i] != 0:
                    di_plus[i] = 100 * dm_plus_sum[i] / tr_sum[i]
                    di_minus[i] = 100 * dm_minus_sum[i] / tr_sum[i]
            
            # DX and ADX
            dx = np.full(len(tr_1d), np.nan)
            for i in range(14, len(tr_1d)):
                if di_plus[i] + di_minus[i] != 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
            
            adx_1d = np.full(len(tr_1d), np.nan)
            if len(dx) >= 28:  # Need 14 for DX smoothing + 14 for ADX
                adx_1d[27] = np.nanmean(dx[14:28])  # First ADX value
                for i in range(28, len(dx)):
                    adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Trend filter: ADX > 25 indicates strong trend
    trend_filter_1d = adx_1d > 25
    
    # Align to 4h timeframe
    trend_filter_aligned = align_htf_to_ltf(prices, df_1d, trend_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_filter_aligned[i]):
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
            # Exit: price closes below Donchian lower OR weak trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                not trend_filter_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR weak trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                not trend_filter_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + strong trend + volume spike
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with strong trend and volume
                if bull_breakout and trend_filter_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with strong trend and volume
                elif bear_breakout and trend_filter_aligned[i] and volume_filter:
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