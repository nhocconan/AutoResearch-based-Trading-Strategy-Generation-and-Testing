#!/usr/bin/env python3
"""
12h 1-Day Trend + Weekly Volatility Breakout
Hypothesis: Combines daily trend alignment with weekly volatility breakouts to capture institutional momentum
in both bull and bear markets. Uses weekly ATR-based breakout levels to avoid whipsaw and volume confirmation
to filter low-quality signals. Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_trend_weekly_vol_breakout_v1"
timeframe = "12h"
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
    
    # 14-period ATR for stop loss and volatility
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
    
    # Daily EMA50 for trend bias (from 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Weekly ATR for breakout levels (from 1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR (14-period)
    atr_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        tr_1w = np.maximum(
            high_1w[1:] - low_1w[1:],
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
        if len(tr_1w) > 0:
            atr_1w[1] = tr_1w[0]
            for i in range(2, len(close_1w)):
                atr_1w[i] = (tr_1w[i-1] * 13 + atr_1w[i-1]) / 14
    
    # Weekly breakout levels: ±1.5 * ATR from weekly close
    upper_breakout = close_1w + 1.5 * atr_1w
    lower_breakout = close_1w - 1.5 * atr_1w
    
    # Align weekly levels to 12h timeframe
    upper_breakout_aligned = align_htf_to_ltf(prices, df_1w, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_1w, lower_breakout)
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 100  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(upper_breakout_aligned[i]) or np.isnan(lower_breakout_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below weekly lower breakout OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower_breakout_aligned[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above weekly upper breakout OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper_breakout_aligned[i] or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                # Breakout entries: weekly levels with trend alignment
                bull_breakout = close[i] > upper_breakout_aligned[i]
                bear_breakout = close[i] < lower_breakout_aligned[i]
                
                # Long: bullish breakout with uptrend and volume
                if bull_breakout and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with downtrend and volume
                elif bear_breakout and trend_bias_aligned[i] == -1 and volume_filter:
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
12h 1-Day Trend + Weekly Volatility Breakout
Hypothesis: Combines daily trend alignment with weekly volatility breakouts to capture institutional momentum
in both bull and bear markets. Uses weekly ATR-based breakout levels to avoid whipsaw and volume confirmation
to filter low-quality signals. Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_trend_weekly_vol_breakout_v1"
timeframe = "12h"
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
    
    # 14-period ATR for stop loss and volatility
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
    
    # Daily EMA50 for trend bias (from 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Weekly ATR for breakout levels (from 1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR (14-period)
    atr_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        tr_1w = np.maximum(
            high_1w[1:] - low_1w[1:],
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
        if len(tr_1w) > 0:
            atr_1w[1] = tr_1w[0]
            for i in range(2, len(close_1w)):
                atr_1w[i] = (tr_1w[i-1] * 13 + atr_1w[i-1]) / 14
    
    # Weekly breakout levels: ±1.5 * ATR from weekly close
    upper_breakout = close_1w + 1.5 * atr_1w
    lower_breakout = close_1w - 1.5 * atr_1w
    
    # Align weekly levels to 12h timeframe
    upper_breakout_aligned = align_htf_to_ltf(prices, df_1w, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_1w, lower_breakout)
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 100  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(upper_breakout_aligned[i]) or np.isnan(lower_breakout_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below weekly lower breakout OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower_breakout_aligned[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above weekly upper breakout OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper_breakout_aligned[i] or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                # Breakout entries: weekly levels with trend alignment
                bull_breakout = close[i] > upper_breakout_aligned[i]
                bear_breakout = close[i] < lower_breakout_aligned[i]
                
                # Long: bullish breakout with uptrend and volume
                if bull_breakout and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with downtrend and volume
                elif bear_breakout and trend_bias_aligned[i] == -1 and volume_filter:
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