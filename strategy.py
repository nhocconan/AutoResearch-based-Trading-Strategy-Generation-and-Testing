#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with 1d Trend Filter and Volume Confirmation v1
Hypothesis: Camarilla pivot levels on 1d timeframe identify key support/resistance.
Price rejection at R3/S3 levels with 1d trend alignment and volume confirmation
provides high-probability reversal trades. Works in both bull and bear markets
by fading extremes in ranging conditions and following breakouts in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_reversal_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    
    # Support levels
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Daily trend filter using EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_prev = np.roll(ema50_1d, 1)
    ema50_prev[0] = ema50_1d[0]
    ema50_rising = ema50_1d > ema50_prev
    ema50_falling = ema50_1d < ema50_prev
    
    # Align all 1d data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For daily EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ema[i]) or
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite level break or stoploss
        if position == 1:  # long position
            # Exit: price breaks above R4 OR stoploss
            if (close[i] >= r4_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks below S4 OR stoploss
            if (close[i] <= s4_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: rejection at R3/S3 with trend alignment
            # Long: rejection at S3 in uptrend
            long_entry = (close[i] <= s3_aligned[i] * 1.002 and  # near S3
                          low[i] < s3_aligned[i] and            # pierced S3
                          close[i] > s3_aligned[i] and          # closed back above
                          ema50_rising_aligned[i] and           # uptrend
                          volume[i] > vol_ema[i] * 1.5)         # volume confirmation
            
            # Short: rejection at R3 in downtrend
            short_entry = (close[i] >= r3_aligned[i] * 0.998 and # near R3
                           high[i] > r3_aligned[i] and           # pierced R3
                           close[i] < r3_aligned[i] and          # closed back below
                           ema50_falling_aligned[i] and          # downtrend
                           volume[i] > vol_ema[i] * 1.5)         # volume confirmation
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with 1d Trend Filter and Volume Confirmation v1
Hypothesis: Camarilla pivot levels on 1d timeframe identify key support/resistance.
Price rejection at R3/S3 levels with 1d trend alignment and volume confirmation
provides high-probability reversal trades. Works in both bull and bear markets
by fading extremes in ranging conditions and following breakouts in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_reversal_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    
    # Support levels
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Daily trend filter using EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_prev = np.roll(ema50_1d, 1)
    ema50_prev[0] = ema50_1d[0]
    ema50_rising = ema50_1d > ema50_prev
    ema50_falling = ema50_1d < ema50_prev
    
    # Align all 1d data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For daily EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ema[i]) or
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite level break or stoploss
        if position == 1:  # long position
            # Exit: price breaks above R4 OR stoploss
            if (close[i] >= r4_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks below S4 OR stoploss
            if (close[i] <= s4_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: rejection at R3/S3 with trend alignment
            # Long: rejection at S3 in uptrend
            long_entry = (close[i] <= s3_aligned[i] * 1.002 and  # near S3
                          low[i] < s3_aligned[i] and            # pierced S3
                          close[i] > s3_aligned[i] and          # closed back above
                          ema50_rising_aligned[i] and           # uptrend
                          volume[i] > vol_ema[i] * 1.5)         # volume confirmation
            
            # Short: rejection at R3 in downtrend
            short_entry = (close[i] >= r3_aligned[i] * 0.998 and # near R3
                           high[i] > r3_aligned[i] and           # pierced R3
                           close[i] < r3_aligned[i] and          # closed back below
                           ema50_falling_aligned[i] and          # downtrend
                           volume[i] > vol_ema[i] * 1.5)         # volume confirmation
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals