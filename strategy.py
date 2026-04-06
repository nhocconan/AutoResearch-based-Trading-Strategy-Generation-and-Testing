#!/usr/bin/env python3
"""
6h 200-day Trend + Volume Breakout + ATR Stop
Hypothesis: Uses 1d 200-period EMA as trend filter to avoid counter-trend trades, 
combined with 20-period high/low breakouts and volume confirmation. 
This avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_200trend_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
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
    
    # 1d EMA200 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200[i] = (close_1d[i] * 2 + ema_200[i-1] * 18) / 20
    
    # Trend bias: above EMA200 = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_200, 1, -1)
    
    # Align to 6h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # 20-period high/low for breakout
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            high_20[i] = np.max(high[i-19:i+1])
            low_20[i] = np.min(low[i-19:i+1])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 200  # Need EMA200 warmup
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below 20-period low OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < low_20[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above 20-period high OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > high_20[i] or
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
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: only in direction of 1d trend
                bull_breakout = close[i] > high_20[i]
                bear_breakout = close[i] < low_20[i]
                
                # Long: breakout above 20-period high with uptrend and volume
                if bull_breakout and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakout below 20-period low with downtrend and volume
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
6h Volume-Weighted Price Action + Trend Filter
Hypothesis: Combines volume-weighted average price (VWAP) deviation with 
1d trend filter to identify high-probability mean reversion and breakout 
opportunities. Uses volume confirmation to avoid false signals. 
Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vwap_trend_filter_v1"
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
    
    # 1d EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2 + ema_50[i-1] * 18) / 20
    
    # Trend bias: above EMA50 = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_50, 1, -1)
    
    # Align to 6h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # VWAP (20-period)
    vwap = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            typical_price = (high[i] + low[i] + close[i]) / 3.0
            vwap[i] = np.sum(volume[i-19:i+1] * 
                           [(high[j] + low[j] + close[j]) / 3.0 
                            for j in range(i-19, i+1)]) / np.sum(volume[i-19:i+1])
    
    # VWAP deviation bands (1.5 * ATR)
    vwap_upper = np.full(n, np.nan)
    vwap_lower = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            if not np.isnan(vwap[i]) and not np.isnan(atr[i]):
                vwap_upper[i] = vwap[i] + 1.5 * atr[i]
                vwap_lower[i] = vwap[i] - 1.5 * atr[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(vwap_upper[i]) or 
            np.isnan(vwap_lower[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i+1])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses below VWAP OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < vwap[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price crosses above VWAP OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > vwap[i] or
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
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Mean reversion: price at VWAP bands with trend
                near_upper = close[i] >= vwap_upper[i] * 0.999  # Allow small tolerance
                near_lower = close[i] <= vwap_lower[i] * 1.001
                
                # Long: price at lower band in uptrend with volume
                if near_lower and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: price at upper band in downtrend with volume
                elif near_upper and trend_bias_aligned[i] == -1 and volume_filter:
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
6h Institutional Flow Detector + Trend Filter
Hypothesis: Detects institutional activity through volume spikes combined with 
price action at key levels (VWAP and swing points), filtered by 1d trend.
Institutional footprints often precede sustained moves. Works in both bull 
(accumulation dips) and bear (distribution rallies).
Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_institutional_flow_v1"
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
    
    # 1d EMA100 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_100 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 100:
        ema_100[99] = np.mean(close_1d[:100])
        for i in range(100, len(close_1d)):
            ema_100[i] = (close_1d[i] * 2 + ema_100[i-1] * 18) / 20
    
    # Trend bias: above EMA100 = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_100, 1, -1)
    
    # Align to 6h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # VWAP (20-period)
    vwap = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            typical_price = (high[i] + low[i] + close[i]) / 3.0
            vwap[i] = np.sum(volume[i-19:i+1] * 
                           [(high[j] + low[j] + close[j]) / 3.0 
                            for j in range(i-19, i+1)]) / np.sum(volume[i-19:i+1])
    
    # Swing high/low (5-period)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    if n >= 5:
        for i in range(4, n):
            swing_high[i] = np.max(high[i-4:i+1])
            swing_low[i] = np.min(low[i-4:i+1])
    
    # Volume spike detector (3x 20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(swing_high[i]) or 
            np.isnan(swing_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: spike > 3x average
        volume_spike = volume[i] > vol_ma[i] * 3.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks swing low OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < swing_low[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks swing high OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > swing_high[i] or
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
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Institutional footprint: volume spike at key levels
                at_vwap = abs(close[i] - vwap[i]) < (atr[i] * 0.5)
                at_swing_high = abs(close[i] - swing_high[i]) < (atr[i] * 0.5)
                at_swing_low = abs(close[i] - swing_low[i]) < (atr[i] * 0.5)
                
                # Long: volume spike at VWAP or swing low in uptrend
                if volume_spike and (at_vwap or at_swing_low) and trend_bias_aligned[i] == 1:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: volume spike at VWAP or swing high in downtrend
                elif volume_spike and (at_vwap or at_swing_high) and trend_bias_aligned[i] == -1:
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
6h 50-200 EMA Cross + Volume Filter + ATR Stop
Hypothesis: Uses 50/200 EMA crossover on 1d timeframe for trend direction,
combined with 6s period high/low breakouts and volume confirmation to 
avoid whipsaws. Works in bull (buy on golden cross) and bear (sell on death cross).
Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_cross_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
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
    
    # 1d EMA50 and EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    ema_200 = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2 + ema_50[i-1] * 18) / 20
    
    if len(close_1d) >= 200:
        ema_200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200[i] = (close_1d[i] * 2 + ema_200[i-1] * 18) / 20
    
    # EMA cross signals: 1 = golden cross (bullish), -1 = death cross (bearish), 0 = no cross
    ema_cross = np.zeros(len(close_1d))
    if len(close_1d) >= 200:
        for i in range(200, len(close_1d)):
            if ema_50[i] > ema_200[i] and ema_50[i-1] <= ema_200[i-1]:
                ema_cross[i] = 1  # golden cross
            elif ema_50[i] < ema_200[i] and ema_50[i-1] >= ema_200[i-1]:
                ema_cross[i] = -1  # death cross
    
    # Trend bias from EMA cross (persists until opposite cross)
    trend_bias_1d = np.zeros(len(close_1d))
    current_bias = 0
    for i in range(len(close_1d)):
        if ema_cross[i] != 0:
            current_bias = ema_cross[i]
        trend_bias_1d[i] = current_bias
    
    # Align to 6h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # 20-period high/low for breakout
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            high_20[i] = np.max(high[i-19:i+1])
            low_20[i] = np.min(low[i-19:i+1])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 200  # Need EMA200 warmup
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below 20-period low OR trend turns bearish
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < low_20[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above 20-period high OR trend turns bullish
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > high_20[i] or
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
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: only in direction of 1d trend
                bull_breakout = close[i] > high_20[i]
                bear_breakout = close[i] < low_20[i]
                
                # Long: breakout above 20-period high with bullish trend and volume
                if bull_breakout and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakout below 20-period low with bearish trend and volume
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
6h Donchian Channel (20) + Volume Confirmation + 1d Trend Filter
Hypothesis: Combines classic Donchian breakouts with volume confirmation and 
1d EMA50 trend filter to avoid false breakouts. Works in bull (buy breakouts 
in uptrend) and bear (sell breakdowns in downtrend). 
Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_volume_trend_v1"
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
    
    # 1d EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2 + ema_50[i-1] * 18) / 20
    
    # Trend bias: above EMA50 = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_50, 1, -1)
    
    # Align to 6h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Donchian Channel (20-period)
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            high_20[i] = np.max(high[i-19:i+1])
            low_20[i] = np.min(low[i-19:i+1])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below 20-period low OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < low_20[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above 20-period high OR against