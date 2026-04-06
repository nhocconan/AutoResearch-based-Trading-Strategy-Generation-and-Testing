#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Trend Filter and Volume Confirmation
Hypothesis: Ichimoku provides robust trend, momentum, and support/resistance levels.
Use 1d trend (price above/below Kumo) for bias and volume confirmation for entry.
Works in bull (buy when price above cloud with bullish TK cross) and bear (sell when price below cloud with bearish TK cross).
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        if i + kijun_period < n:
            senkou_a[i + kijun_period] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        if i + kijun_period < n:
            senkou_b[i + kijun_period] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku Cloud (same parameters)
    tenkan_1d = np.full(len(close_1d), np.nan)
    kijun_1d = np.full(len(close_1d), np.nan)
    senkou_a_1d = np.full(len(close_1d), np.nan)
    senkou_b_1d = np.full(len(close_1d), np.nan)
    
    for i in range(tenkan_period - 1, len(close_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-tenkan_period+1:i+1]) + np.min(low_1d[i-tenkan_period+1:i+1])) / 2
    
    for i in range(kijun_period - 1, len(close_1d)):
        kijun_1d[i] = (np.max(high_1d[i-kijun_period+1:i+1]) + np.min(low_1d[i-kijun_period+1:i+1])) / 2
    
    for i in range(kijun_period - 1, len(close_1d)):
        if i + kijun_period < len(close_1d):
            senkou_a_1d[i + kijun_period] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    for i in range(senkou_span_b_period - 1, len(close_1d)):
        if i + kijun_period < len(close_1d):
            senkou_b_1d[i + kijun_period] = (np.max(high_1d[i-senkou_span_b_period+1:i+1]) + np.min(low_1d[i-senkou_span_b_period+1:i+1])) / 2
    
    # 1d Trend: price above/both Senkou spans = bullish, price below/both = bearish
    trend_1d = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if not np.isnan(senkou_a_1d[i]) and not np.isnan(senkou_b_1d[i]):
            if close_1d[i] > senkou_a_1d[i] and close_1d[i] > senkou_b_1d[i]:
                trend_1d[i] = 1  # bullish
            elif close_1d[i] < senkou_a_1d[i] and close_1d[i] < senkou_b_1d[i]:
                trend_1d[i] = -1  # bearish
            else:
                trend_1d[i] = 0  # neutral (in cloud)
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):  # 20-period MA
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need enough for Ichimoku calculations)
    start = senkou_span_b_period + kijun_period  # ~78
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
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
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses below Kijun OR trend turns bearish
            # Stoploss: price drops 2*ATR below entry (using 26-period ATR approximation)
            if (close[i] < kijun[i] or
                trend_1d_aligned[i] == -1 or
                (i >= 26 and close[i] < close[i-26] - 2.0 * (np.max(high[i-25:i+1]) - np.min(low[i-25:i+1])))):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price crosses above Kijun OR trend turns bullish
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > kijun[i] or
                trend_1d_aligned[i] == 1 or
                (i >= 26 and close[i] > close[i-26] + 2.0 * (np.max(high[i-25:i+1]) - np.min(low[i-25:i+1])))):
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
                # TK Cross signals
                tk_cross_bull = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
                tk_cross_bear = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
                
                # Cloud twist (Senkou A/B cross) for stronger signals
                senkou_a_above_b = senkou_a[i] > senkou_b[i]
                senkou_b_above_a = senkou_b[i] > senkou_a[i]
                
                # Long: bullish TK cross above cloud with bullish 1d trend + volume
                if (tk_cross_bull and senkou_a_above_b and 
                    close[i] > senkou_a[i] and close[i] > senkou_b[i] and
                    trend_1d_aligned[i] == 1 and volume_filter):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish TK cross below cloud with bearish 1d trend + volume
                elif (tk_cross_bear and senkou_b_above_a and 
                      close[i] < senkou_a[i] and close[i] < senkou_b[i] and
                      trend_1d_aligned[i] == -1 and volume_filter):
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
6h Ichimoku Cloud with 1d Trend Filter and Volume Confirmation
Hypothesis: Ichimoku provides robust trend, momentum, and support/resistance levels.
Use 1d trend (price above/below Kumo) for bias and volume confirmation for entry.
Works in bull (buy when price above cloud with bullish TK cross) and bear (sell when price below cloud with bearish TK cross).
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        if i + kijun_period < n:
            senkou_a[i + kijun_period] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        if i + kijun_period < n:
            senkou_b[i + kijun_period] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku Cloud (same parameters)
    tenkan_1d = np.full(len(close_1d), np.nan)
    kijun_1d = np.full(len(close_1d), np.nan)
    senkou_a_1d = np.full(len(close_1d), np.nan)
    senkou_b_1d = np.full(len(close_1d), np.nan)
    
    for i in range(tenkan_period - 1, len(close_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-tenkan_period+1:i+1]) + np.min(low_1d[i-tenkan_period+1:i+1])) / 2
    
    for i in range(kijun_period - 1, len(close_1d)):
        kijun_1d[i] = (np.max(high_1d[i-kijun_period+1:i+1]) + np.min(low_1d[i-kijun_period+1:i+1])) / 2
    
    for i in range(kijun_period - 1, len(close_1d)):
        if i + kijun_period < len(close_1d):
            senkou_a_1d[i + kijun_period] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    for i in range(senkou_span_b_period - 1, len(close_1d)):
        if i + kijun_period < len(close_1d):
            senkou_b_1d[i + kijun_period] = (np.max(high_1d[i-senkou_span_b_period+1:i+1]) + np.min(low_1d[i-senkou_span_b_period+1:i+1])) / 2
    
    # 1d Trend: price above/both Senkou spans = bullish, price below/both = bearish
    trend_1d = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if not np.isnan(senkou_a_1d[i]) and not np.isnan(senkou_b_1d[i]):
            if close_1d[i] > senkou_a_1d[i] and close_1d[i] > senkou_b_1d[i]:
                trend_1d[i] = 1  # bullish
            elif close_1d[i] < senkou_a_1d[i] and close_1d[i] < senkou_b_1d[i]:
                trend_1d[i] = -1  # bearish
            else:
                trend_1d[i] = 0  # neutral (in cloud)
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):  # 20-period MA
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need enough for Ichimoku calculations)
    start = senkou_span_b_period + kijun_period  # ~78
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
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
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses below Kijun OR trend turns bearish
            # Stoploss: price drops 2*ATR below entry (using 26-period ATR approximation)
            if (close[i] < kijun[i] or
                trend_1d_aligned[i] == -1 or
                (i >= 26 and close[i] < close[i-26] - 2.0 * (np.max(high[i-25:i+1]) - np.min(low[i-25:i+1])))):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price crosses above Kijun OR trend turns bullish
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > kijun[i] or
                trend_1d_aligned[i] == 1 or
                (i >= 26 and close[i] > close[i-26] + 2.0 * (np.max(high[i-25:i+1]) - np.min(low[i-25:i+1])))):
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
                # TK Cross signals
                tk_cross_bull = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
                tk_cross_bear = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
                
                # Cloud twist (Senkou A/B cross) for stronger signals
                senkou_a_above_b = senkou_a[i] > senkou_b[i]
                senkou_b_above_a = senkou_b[i] > senkou_a[i]
                
                # Long: bullish TK cross above cloud with bullish 1d trend + volume
                if (tk_cross_bull and senkou_a_above_b and 
                    close[i] > senkou_a[i] and close[i] > senkou_b[i] and
                    trend_1d_aligned[i] == 1 and volume_filter):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish TK cross below cloud with bearish 1d trend + volume
                elif (tk_cross_bear and senkou_b_above_a and 
                      close[i] < senkou_a[i] and close[i] < senkou_b[i] and
                      trend_1d_aligned[i] == -1 and volume_filter):
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