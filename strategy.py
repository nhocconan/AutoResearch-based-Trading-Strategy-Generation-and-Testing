#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume Spike + ATR Filter
Hypothesis: Camarilla pivot levels provide strong support/resistance at key intraday levels.
Fade at R3/S3 (mean reversion) and breakout continuation at R4/S4 (trend following).
Volume spike confirms institutional participation. ATR filter ensures volatility regime.
Works in bull (buy R3 bounce, sell R4 break) and bear (sell S3 bounce, buy S4 break).
Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee decay.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_v1"
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
    
    # Get 1d data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 2:
        for i in range(1, len(close_1d)):
            # Previous day's OHLC
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            # Camarilla formulas
            range_val = ph - pl
            camarilla_r4[i] = pc + range_val * 1.500
            camarilla_r3[i] = pc + range_val * 1.250
            camarilla_s3[i] = pc - range_val * 1.250
            camarilla_s4[i] = pc - range_val * 1.500
    
    # Align Camarilla levels to 6h timeframe (previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For volume average and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: reverse signal or stoploss
            if (close[i] < s3_aligned[i] or  # Reverse at S3
                close[i] < entry_price - 2.0 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: reverse signal or stoploss
            if (close[i] > r3_aligned[i] or  # Reverse at R3
                close[i] > entry_price + 2.0 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Fade at R3/S3 (mean reversion)
            fade_r3 = close[i] < r3_aligned[i] and close[i] > s3_aligned[i]
            fade_s3 = close[i] > s3_aligned[i] and close[i] < r3_aligned[i]
            
            # Breakout continuation at R4/S4 (trend following)
            breakout_r4 = close[i] > r4_aligned[i]
            breakdown_s4 = close[i] < s4_aligned[i]
            
            if fade_r3 and volume_filter:
                # Fade R3: expect pullback to S3, go long
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif fade_s3 and volume_filter:
                # Fade S3: expect pullback to R3, go short
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            elif breakout_r4 and volume_filter:
                # Breakout R4: continuation upward
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif breakdown_s4 and volume_filter:
                # Breakdown S4: continuation downward
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume Spike + ATR Filter
Hypothesis: Camarilla pivot levels provide strong support/resistance at key intraday levels.
Fade at R3/S3 (mean reversion) and breakout continuation at R4/S4 (trend following).
Volume spike confirms institutional participation. ATR filter ensures volatility regime.
Works in bull (buy R3 bounce, sell R4 break) and bear (sell S3 bounce, buy S4 break).
Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee decay.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_v1"
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
    
    # Get 1d data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 2:
        for i in range(1, len(close_1d)):
            # Previous day's OHLC
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            # Camarilla formulas
            range_val = ph - pl
            camarilla_r4[i] = pc + range_val * 1.500
            camarilla_r3[i] = pc + range_val * 1.250
            camarilla_s3[i] = pc - range_val * 1.250
            camarilla_s4[i] = pc - range_val * 1.500
    
    # Align Camarilla levels to 6h timeframe (previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For volume average and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: reverse signal or stoploss
            if (close[i] < s3_aligned[i] or  # Reverse at S3
                close[i] < entry_price - 2.0 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: reverse signal or stoploss
            if (close[i] > r3_aligned[i] or  # Reverse at R3
                close[i] > entry_price + 2.0 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Fade at R3/S3 (mean reversion)
            fade_r3 = close[i] < r3_aligned[i] and close[i] > s3_aligned[i]
            fade_s3 = close[i] > s3_aligned[i] and close[i] < r3_aligned[i]
            
            # Breakout continuation at R4/S4 (trend following)
            breakout_r4 = close[i] > r4_aligned[i]
            breakdown_s4 = close[i] < s4_aligned[i]
            
            if fade_r3 and volume_filter:
                # Fade R3: expect pullback to S3, go long
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif fade_s3 and volume_filter:
                # Fade S3: expect pullback to R3, go short
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            elif breakout_r4 and volume_filter:
                # Breakout R4: continuation upward
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif breakdown_s4 and volume_filter:
                # Breakdown S4: continuation downward
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>