#!/usr/bin/env python3
"""
6h Camarilla Pivot Breakout with Volume Confirmation
Hypothesis: Camarilla pivot levels (derived from previous 1d) act as strong support/resistance.
Breakouts above R4 or below S4 with volume confirmation indicate institutional interest and
continuation of momentum. In ranging markets, reversals at R3/S3 provide mean-reversion
opportunities. Works in both bull and bear markets by adapting to volatility regimes.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_vol"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # Typical price for pivot
    typical_price = (prev_high + prev_low + prev_close) / 3
    # Camarilla levels
    r4 = typical_price + ((prev_high - prev_low) * 1.1 / 2)
    r3 = typical_price + ((prev_high - prev_low) * 1.1 / 4)
    s3 = typical_price - ((prev_high - prev_low) * 1.1 / 4)
    s4 = typical_price - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align pivots to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or \
           np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below S3 (mean reversion) OR stoploss hit
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s3_6h[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R3 (mean reversion) OR stoploss hit
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r3_6h[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above R4 with volume (bullish continuation)
            if (close[i] > r4_6h[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below S4 with volume (bearish continuation)
            elif (close[i] < s4_6h[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
"""
6h Camarilla Pivot Breakout with Volume Confirmation
Hypothesis: Camarilla pivot levels (derived from previous 1d) act as strong support/resistance.
Breakouts above R4 or below S4 with volume confirmation indicate institutional interest and
continuation of momentum. In ranging markets, reversals at R3/S3 provide mean-reversion
opportunities. Works in both bull and bear markets by adapting to volatility regimes.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_vol"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # Typical price for pivot
    typical_price = (prev_high + prev_low + prev_close) / 3
    # Camarilla levels
    r4 = typical_price + ((prev_high - prev_low) * 1.1 / 2)
    r3 = typical_price + ((prev_high - prev_low) * 1.1 / 4)
    s3 = typical_price - ((prev_high - prev_low) * 1.1 / 4)
    s4 = typical_price - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align pivots to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or \
           np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below S3 (mean reversion) OR stoploss hit
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s3_6h[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R3 (mean reversion) OR stoploss hit
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r3_6h[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above R4 with volume (bullish continuation)
            if (close[i] > r4_6h[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below S4 with volume (bearish continuation)
            elif (close[i] < s4_6h[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>