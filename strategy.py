#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d trend filter and volume confirmation
Hypothesis: Ichimoku provides dynamic support/resistance and trend direction. Using daily timeframe for trend filter ensures alignment with higher timeframe momentum, while volume confirmation filters false signals. Works in bull (buy when price above cloud with bullish TK cross) and bear (sell when price below cloud with bearish TK cross). Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 6:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 5 + atr[i-1]) / 6
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on daily close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # Daily trend: above EMA50 = bullish, below = bearish
    daily_trend = np.where(close_1d > ema_1d, 1, -1)
    
    # Align daily trend to 6h timeframe
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on daily
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = np.full(n, np.nan)
    for i in range(8, n):
        tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = np.full(n, np.nan)
    for i in range(25, n):
        kijun[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    for i in range(26, n):
        senkou_a[i] = (tenkan[i-26] + kijun[i-26]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
    senkou_b = np.full(n, np.nan)
    for i in range(51, n):
        senkou_b[i] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou = np.full(n, np.nan)
    for i in range(26, n):
        chikou[i] = close[i-26]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 78  # Need enough data for Ichimoku (52+26)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(daily_trend_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(chikou[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x daily average volume (scaled)
        # Scale daily volume to 6h: approx 1/4 of daily volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price falls below cloud bottom OR against daily trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < cloud_bottom or
                daily_trend_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above cloud top OR against daily trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > cloud_top or
                daily_trend_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 24 bars flat
            if bars_since_entry >= 24:
                # Entry conditions
                # Bullish: price above cloud, TK cross bullish, chikou above price 26 periods ago
                bullish = (close[i] > cloud_top and 
                          tenkan[i] > kijun[i] and 
                          chikou[i] > close[i])
                
                # Bearish: price below cloud, TK cross bearish, chikou below price 26 periods ago
                bearish = (close[i] < cloud_bottom and 
                          tenkan[i] < kijun[i] and 
                          chikou[i] < close[i])
                
                # Long: bullish conditions with bullish daily trend + volume
                if bullish and daily_trend_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish conditions with bearish daily trend + volume
                elif bearish and daily_trend_aligned[i] == -1 and volume_filter:
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
6h Ichimoku Cloud with 1d trend filter and volume confirmation
Hypothesis: Ichimoku provides dynamic support/resistance and trend direction. Using daily timeframe for trend filter ensures alignment with higher timeframe momentum, while volume confirmation filters false signals. Works in bull (buy when price above cloud with bullish TK cross) and bear (sell when price below cloud with bearish TK cross). Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 6:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 5 + atr[i-1]) / 6
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on daily close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # Daily trend: above EMA50 = bullish, below = bearish
    daily_trend = np.where(close_1d > ema_1d, 1, -1)
    
    # Align daily trend to 6h timeframe
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on daily
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = np.full(n, np.nan)
    for i in range(8, n):
        tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = np.full(n, np.nan)
    for i in range(25, n):
        kijun[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    for i in range(26, n):
        senkou_a[i] = (tenkan[i-26] + kijun[i-26]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
    senkou_b = np.full(n, np.nan)
    for i in range(51, n):
        senkou_b[i] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou = np.full(n, np.nan)
    for i in range(26, n):
        chikou[i] = close[i-26]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 78  # Need enough data for Ichimoku (52+26)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(daily_trend_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(chikou[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x daily average volume (scaled)
        # Scale daily volume to 6h: approx 1/4 of daily volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price falls below cloud bottom OR against daily trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < cloud_bottom or
                daily_trend_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above cloud top OR against daily trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > cloud_top or
                daily_trend_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 24 bars flat
            if bars_since_entry >= 24:
                # Entry conditions
                # Bullish: price above cloud, TK cross bullish, chikou above price 26 periods ago
                bullish = (close[i] > cloud_top and 
                          tenkan[i] > kijun[i] and 
                          chikou[i] > close[i])
                
                # Bearish: price below cloud, TK cross bearish, chikou below price 26 periods ago
                bearish = (close[i] < cloud_bottom and 
                          tenkan[i] < kijun[i] and 
                          chikou[i] < close[i])
                
                # Long: bullish conditions with bullish daily trend + volume
                if bullish and daily_trend_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish conditions with bearish daily trend + volume
                elif bearish and daily_trend_aligned[i] == -1 and volume_filter:
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