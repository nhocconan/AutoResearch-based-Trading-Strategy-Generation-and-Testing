#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Trend Filter + Volume Spike
Hypothesis: Uses Donchian breakouts on 6h with 1d EMA50 trend filter and volume confirmation.
Captures momentum in trending markets while avoiding chop. Works in both bull and bear
by taking breakouts in direction of higher timeframe trend. Targets 15-25 trades/year
to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1dtrend_vol_v2"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # Trend filter: above EMA50 = bullish, below = bearish
    trend_filter_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align to 6h timeframe
    trend_filter_aligned = align_htf_to_ltf(prices, df_1d, trend_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # bars since last exit to enforce minimum hold
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_filter_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                trend_filter_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                trend_filter_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + 1d trend + volume + minimum hold
            # Minimum 10 bars between entries to reduce churn
            if bars_since_exit >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with bullish 1d trend and volume
                if bull_breakout and trend_filter_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: bearish breakout with bearish 1d trend and volume
                elif bear_breakout and trend_filter_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Trend Filter + Volume Spike
Hypothesis: Uses Donchian breakouts on 6h with 1d EMA50 trend filter and volume confirmation.
Captures momentum in trending markets while avoiding chop. Works in both bull and bear
by taking breakouts in direction of higher timeframe trend. Targets 15-25 trades/year
to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1dtrend_vol_v2"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # Trend filter: above EMA50 = bullish, below = bearish
    trend_filter_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align to 6h timeframe
    trend_filter_aligned = align_htf_to_ltf(prices, df_1d, trend_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # bars since last exit to enforce minimum hold
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_filter_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                trend_filter_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                trend_filter_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + 1d trend + volume + minimum hold
            # Minimum 10 bars between entries to reduce churn
            if bars_since_exit >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with bullish 1d trend and volume
                if bull_breakout and trend_filter_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: bearish breakout with bearish 1d trend and volume
                elif bear_breakout and trend_filter_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals

</think>