#!/usr/bin/env python3
"""
4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
Hypothesis: 4h Donchian breakouts capture intermediate-term momentum. Filter by 1d ADX > 25 for trend strength and volume confirmation for conviction. Works in trending markets (both bull and bear) by taking breakouts in the direction of the 1d trend. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_adx_vol_v2"
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
    
    # Get 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation on 1d data
    adx = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        # True Range
        tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
        tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
        tr1 = np.concatenate([[np.nan], tr1])
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values
        tr14 = np.full(len(close_1d), np.nan)
        plus_dm14 = np.full(len(close_1d), np.nan)
        minus_dm14 = np.full(len(close_1d), np.nan)
        
        if len(close_1d) >= 14:
            tr14[13] = np.nansum(tr1[1:15])
            plus_dm14[13] = np.nansum(plus_dm[1:15])
            minus_dm14[13] = np.nansum(minus_dm[1:15])
            
            for i in range(14, len(close_1d)):
                tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr1[i]
                plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / 14) + plus_dm[i]
                minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / 14) + minus_dm[i]
        
        # Directional Indicators
        plus_di = np.full(len(close_1d), np.nan)
        minus_di = np.full(len(close_1d), np.nan)
        dx = np.full(len(close_1d), np.nan)
        
        for i in range(13, len(close_1d)):
            if not np.isnan(tr14[i]) and tr14[i] != 0:
                plus_di[i] = (plus_dm14[i] / tr14[i]) * 100
                minus_di[i] = (minus_dm14[i] / tr14[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX (smoothed DX)
        if len(close_1d) >= 27:
            adx[27] = np.nanmean(dx[14:28])
            for i in range(28, len(close_1d)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Trend direction from 1d close vs 50 EMA
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    
    # Align 1d indicators to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) from 4h data
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
            np.isnan(trend_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_1d_aligned[i] > 25
        
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 4h: approx 1/6 of 1d volume (since 6x 4h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 6.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: upper/lower with 1d trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish 1d trend + ADX + volume
                if bull_breakout and trend_1d_aligned[i] == 1 and trend_filter and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish 1d trend + ADX + volume
                elif bear_breakout and trend_1d_aligned[i] == -1 and trend_filter and volume_filter:
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
4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
Hypothesis: 4h Donchian breakouts capture intermediate-term momentum. Filter by 1d ADX > 25 for trend strength and volume confirmation for conviction. Works in trending markets (both bull and bear) by taking breakouts in the direction of the 1d trend. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_adx_vol_v2"
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
    
    # Get 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation on 1d data
    adx = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        # True Range
        tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
        tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
        tr1 = np.concatenate([[np.nan], tr1])
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values
        tr14 = np.full(len(close_1d), np.nan)
        plus_dm14 = np.full(len(close_1d), np.nan)
        minus_dm14 = np.full(len(close_1d), np.nan)
        
        if len(close_1d) >= 14:
            tr14[13] = np.nansum(tr1[1:15])
            plus_dm14[13] = np.nansum(plus_dm[1:15])
            minus_dm14[13] = np.nansum(minus_dm[1:15])
            
            for i in range(14, len(close_1d)):
                tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr1[i]
                plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / 14) + plus_dm[i]
                minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / 14) + minus_dm[i]
        
        # Directional Indicators
        plus_di = np.full(len(close_1d), np.nan)
        minus_di = np.full(len(close_1d), np.nan)
        dx = np.full(len(close_1d), np.nan)
        
        for i in range(13, len(close_1d)):
            if not np.isnan(tr14[i]) and tr14[i] != 0:
                plus_di[i] = (plus_dm14[i] / tr14[i]) * 100
                minus_di[i] = (minus_dm14[i] / tr14[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX (smoothed DX)
        if len(close_1d) >= 27:
            adx[27] = np.nanmean(dx[14:28])
            for i in range(28, len(close_1d)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Trend direction from 1d close vs 50 EMA
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    
    # Align 1d indicators to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) from 4h data
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
            np.isnan(trend_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_1d_aligned[i] > 25
        
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 4h: approx 1/6 of 1d volume (since 6x 4h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 6.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: upper/lower with 1d trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish 1d trend + ADX + volume
                if bull_breakout and trend_1d_aligned[i] == 1 and trend_filter and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish 1d trend + ADX + volume
                elif bear_breakout and trend_1d_aligned[i] == -1 and trend_filter and volume_filter:
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