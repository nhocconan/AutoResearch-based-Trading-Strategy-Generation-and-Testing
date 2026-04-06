#!/usr/bin/env python3
"""
6h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
Hypothesis: Donchian breakouts capture momentum, filtered by 12h ADX>25 for trending markets and volume confirmation to avoid false breakouts. Works in bull (buy breakouts above) and bear (sell breakdowns below). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX calculation on 12h data
    adx = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 30:  # Need enough data for ADX
        # Calculate +DM and -DM
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range
        tr_12h = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (14-period)
        tr_sum = np.full(len(tr_12h), np.nan)
        plus_dm_sum = np.full(len(plus_dm), np.nan)
        minus_dm_sum = np.full(len(minus_dm), np.nan)
        
        if len(tr_12h) >= 14:
            tr_sum[13] = np.sum(tr_12h[:14])
            plus_dm_sum[13] = np.sum(plus_dm[:14])
            minus_dm_sum[13] = np.sum(minus_dm[:14])
            
            for i in range(14, len(tr_12h)):
                tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / 14) + tr_12h[i]
                plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / 14) + plus_dm[i]
                minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / 14) + minus_dm[i]
        
        # Calculate +DI and -DI
        plus_di = np.full(len(close_12h), np.nan)
        minus_di = np.full(len(close_12h), np.nan)
        if len(tr_sum) > 13 and not np.isnan(tr_sum[13]):
            for i in range(13, len(tr_sum)):
                if tr_sum[i] != 0:
                    plus_di[i] = 100 * (plus_dm_sum[i] / tr_sum[i])
                    minus_di[i] = 100 * (minus_dm_sum[i] / tr_sum[i])
        
        # Calculate DX and ADX
        dx = np.full(len(close_12h), np.nan)
        if len(plus_di) > 13 and not np.isnan(plus_di[13]):
            for i in range(13, len(plus_di)):
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX is smoothed DX
        adx_smooth = np.full(len(dx), np.nan)
        if len(dx) >= 27:  # Need 14+13 for ADX
            adx_smooth[26] = np.nanmean(dx[13:27])  # First ADX value
            for i in range(27, len(dx)):
                adx_smooth[i] = (adx_smooth[i-1] * 13 + dx[i]) / 14
            adx = adx_smooth
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Get 12h data for volume confirmation
    volume_12h = df_12h['volume'].values
    
    # 20-period average volume on 12h
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
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
    start = 60  # Need enough data for all indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 12h average volume (scaled)
        # Scale 12h volume to 6h: approx 1/2 of 12h volume (since 2x 6h in 12h)
        vol_threshold = vol_ma_12h_aligned[i] / 2.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_12h_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR ADX < 20 (trend weakening)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                adx_12h_aligned[i] < 20 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR ADX < 20 (trend weakening)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                adx_12h_aligned[i] < 20 or
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
                # Breakout entries: upper/lower with ADX > 25 and volume
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with ADX > 25 + volume
                if bull_breakout and trend_filter and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with ADX > 25 + volume
                elif bear_breakout and trend_filter and volume_filter:
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
6h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
Hypothesis: Donchian breakouts capture momentum, filtered by 12h ADX>25 for trending markets and volume confirmation to avoid false breakouts. Works in bull (buy breakouts above) and bear (sell breakdowns below). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX calculation on 12h data
    adx = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 30:  # Need enough data for ADX
        # Calculate +DM and -DM
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range
        tr_12h = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (14-period)
        tr_sum = np.full(len(tr_12h), np.nan)
        plus_dm_sum = np.full(len(plus_dm), np.nan)
        minus_dm_sum = np.full(len(minus_dm), np.nan)
        
        if len(tr_12h) >= 14:
            tr_sum[13] = np.sum(tr_12h[:14])
            plus_dm_sum[13] = np.sum(plus_dm[:14])
            minus_dm_sum[13] = np.sum(minus_dm[:14])
            
            for i in range(14, len(tr_12h)):
                tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / 14) + tr_12h[i]
                plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / 14) + plus_dm[i]
                minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / 14) + minus_dm[i]
        
        # Calculate +DI and -DI
        plus_di = np.full(len(close_12h), np.nan)
        minus_di = np.full(len(close_12h), np.nan)
        if len(tr_sum) > 13 and not np.isnan(tr_sum[13]):
            for i in range(13, len(tr_sum)):
                if tr_sum[i] != 0:
                    plus_di[i] = 100 * (plus_dm_sum[i] / tr_sum[i])
                    minus_di[i] = 100 * (minus_dm_sum[i] / tr_sum[i])
        
        # Calculate DX and ADX
        dx = np.full(len(close_12h), np.nan)
        if len(plus_di) > 13 and not np.isnan(plus_di[13]):
            for i in range(13, len(plus_di)):
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX is smoothed DX
        adx_smooth = np.full(len(dx), np.nan)
        if len(dx) >= 27:  # Need 14+13 for ADX
            adx_smooth[26] = np.nanmean(dx[13:27])  # First ADX value
            for i in range(27, len(dx)):
                adx_smooth[i] = (adx_smooth[i-1] * 13 + dx[i]) / 14
            adx = adx_smooth
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Get 12h data for volume confirmation
    volume_12h = df_12h['volume'].values
    
    # 20-period average volume on 12h
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
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
    start = 60  # Need enough data for all indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 12h average volume (scaled)
        # Scale 12h volume to 6h: approx 1/2 of 12h volume (since 2x 6h in 12h)
        vol_threshold = vol_ma_12h_aligned[i] / 2.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_12h_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR ADX < 20 (trend weakening)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                adx_12h_aligned[i] < 20 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR ADX < 20 (trend weakening)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                adx_12h_aligned[i] < 20 or
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
                # Breakout entries: upper/lower with ADX > 25 and volume
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with ADX > 25 + volume
                if bull_breakout and trend_filter and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with ADX > 25 + volume
                elif bear_breakout and trend_filter and volume_filter:
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