#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Confirmation + Trend Filter
Hypothesis: Uses weekly pivot points (from 1w) for major support/resistance with volume confirmation.
Trades breakouts at R4/S4 with weekly trend filter, and mean reversion at R3/S3 against weekly trend.
Weekly timeframe provides structural bias that works in both bull (buy dips to S3 in uptrend) and bear (sell rallies to R3 in downtrend).
Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weeklypivot_volume_v2"
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
    
    # 1w data for weekly pivot and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 18) / 20
    
    # Weekly trend: above EMA = bullish, below = bearish
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    
    # Calculate weekly pivot from previous week's data
    pivot_1w = np.full_like(close_1w, np.nan)
    r1_1w = np.full_like(close_1w, np.nan)
    s1_1w = np.full_like(close_1w, np.nan)
    r2_1w = np.full_like(close_1w, np.nan)
    s2_1w = np.full_like(close_1w, np.nan)
    r3_1w = np.full_like(close_1w, np.nan)
    s3_1w = np.full_like(close_1w, np.nan)
    r4_1w = np.full_like(close_1w, np.nan)
    s4_1w = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
            p = (ph + pl + pc) / 3.0
            pivot_1w[i] = p
            r1_1w[i] = 2*p - pl
            s1_1w[i] = 2*p - ph
            r2_1w[i] = p + (ph - pl)
            s2_1w[i] = p - (ph - pl)
            r3_1w[i] = ph + 2*(p - pl)
            s3_1w[i] = pl - 2*(ph - p)
            r4_1w[i] = 3*p - 2*pl
            s4_1w[i] = 3*ph - 2*pl
    
    # Align weekly data to 6h timeframe
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below S3 (mean reversion) OR against weekly trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s3_1w_aligned[i] or
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above R3 (mean reversion) OR against weekly trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r3_1w_aligned[i] or
                trend_1w_aligned[i] == 1 or
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
                # Breakout entries: R4/S4 with weekly trend
                bull_breakout = close[i] > r4_1w_aligned[i]
                bear_breakout = close[i] < s4_1w_aligned[i]
                
                # Mean reversion entries: R3/S3 against weekly trend (fade)
                near_pivot = abs(close[i] - pivot_1w_aligned[i]) < (r1_1w_aligned[i] - s1_1w_aligned[i]) * 0.5
                
                # Long: breakout with trend OR mean reversion at S3 against downtrend
                if (bull_breakout and trend_1w_aligned[i] == 1 and volume_filter) or \
                   (close[i] > s3_1w_aligned[i] and close[i] < pivot_1w_aligned[i] and 
                    near_pivot and volume_filter and trend_1w_aligned[i] == -1):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown with trend OR mean reversion at R3 against uptrend
                elif (bear_breakout and trend_1w_aligned[i] == -1 and volume_filter) or \
                     (close[i] < r3_1w_aligned[i] and close[i] > pivot_1w_aligned[i] and 
                      near_pivot and volume_filter and trend_1w_aligned[i] == 1):
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
6h Weekly Pivot + Volume Confirmation + Trend Filter
Hypothesis: Uses weekly pivot points (from 1w) for major support/resistance with volume confirmation.
Trades breakouts at R4/S4 with weekly trend filter, and mean reversion at R3/S3 against weekly trend.
Weekly timeframe provides structural bias that works in both bull (buy dips to S3 in uptrend) and bear (sell rallies to R3 in downtrend).
Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weeklypivot_volume_v2"
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
    
    # 1w data for weekly pivot and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 18) / 20
    
    # Weekly trend: above EMA = bullish, below = bearish
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    
    # Calculate weekly pivot from previous week's data
    pivot_1w = np.full_like(close_1w, np.nan)
    r1_1w = np.full_like(close_1w, np.nan)
    s1_1w = np.full_like(close_1w, np.nan)
    r2_1w = np.full_like(close_1w, np.nan)
    s2_1w = np.full_like(close_1w, np.nan)
    r3_1w = np.full_like(close_1w, np.nan)
    s3_1w = np.full_like(close_1w, np.nan)
    r4_1w = np.full_like(close_1w, np.nan)
    s4_1w = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
            p = (ph + pl + pc) / 3.0
            pivot_1w[i] = p
            r1_1w[i] = 2*p - pl
            s1_1w[i] = 2*p - ph
            r2_1w[i] = p + (ph - pl)
            s2_1w[i] = p - (ph - pl)
            r3_1w[i] = ph + 2*(p - pl)
            s3_1w[i] = pl - 2*(ph - p)
            r4_1w[i] = 3*p - 2*pl
            s4_1w[i] = 3*ph - 2*pl
    
    # Align weekly data to 6h timeframe
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below S3 (mean reversion) OR against weekly trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s3_1w_aligned[i] or
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above R3 (mean reversion) OR against weekly trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r3_1w_aligned[i] or
                trend_1w_aligned[i] == 1 or
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
                # Breakout entries: R4/S4 with weekly trend
                bull_breakout = close[i] > r4_1w_aligned[i]
                bear_breakout = close[i] < s4_1w_aligned[i]
                
                # Mean reversion entries: R3/S3 against weekly trend (fade)
                near_pivot = abs(close[i] - pivot_1w_aligned[i]) < (r1_1w_aligned[i] - s1_1w_aligned[i]) * 0.5
                
                # Long: breakout with trend OR mean reversion at S3 against downtrend
                if (bull_breakout and trend_1w_aligned[i] == 1 and volume_filter) or \
                   (close[i] > s3_1w_aligned[i] and close[i] < pivot_1w_aligned[i] and 
                    near_pivot and volume_filter and trend_1w_aligned[i] == -1):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown with trend OR mean reversion at R3 against uptrend
                elif (bear_breakout and trend_1w_aligned[i] == -1 and volume_filter) or \
                     (close[i] < r3_1w_aligned[i] and close[i] > pivot_1w_aligned[i] and 
                      near_pivot and volume_filter and trend_1w_aligned[i] == 1):
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