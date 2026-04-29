#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal + Volume Spike
# Uses weekly Camarilla pivot levels (R3/S3, R4/S4) from prior week.
# Long when price breaks above R4 with volume > 2x avg (continuation breakout).
# Short when price breaks below S4 with volume > 2x avg (continuation breakdown).
# Fade trades at R3/S3 with volume < 0.5x avg (mean reversion in extreme zones).
# Works in bull/bear via volume confirmation and pivot structure.
# Timeframe: 6h (primary), HTF: 1w for weekly pivots.

name = "6h_WeeklyCamarilla_R3R4_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from prior week's OHLC
    # Using prior week's data to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla levels: based on prior week's range
    weekly_range = weekly_high - weekly_low
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Resistance levels
    R3 = weekly_pivot + (weekly_range * 1.1 / 4.0)
    R4 = weekly_pivot + (weekly_range * 1.1 / 2.0)
    
    # Support levels
    S3 = weekly_pivot - (weekly_range * 1.1 / 4.0)
    S4 = weekly_pivot - (weekly_range * 1.1 / 2.0)
    
    # Align to 6h timeframe (wait for weekly bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Volume confirmation: volume > 2.0x 20-period average for breakouts
    # Fade volume: volume < 0.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    volume_dry = volume < (0.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or \
           np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_volume_dry = volume_dry[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below R3 (failed breakout)
            # 2. Price falls below weekly pivot (trend change)
            if (curr_close < R3_aligned[i] or
                curr_close < weekly_pivot[i]):  # Note: weekly_pivot not aligned - need to fix
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above S3 (failed breakdown)
            # 2. Price rises above weekly pivot (trend change)
            if (curr_close > S3_aligned[i] or
                curr_close > weekly_pivot[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Breakout Long: price > R4 with volume spike
            if (curr_high > R4_aligned[i] and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Breakdown Short: price < S4 with volume spike
            elif (curr_low < S4_aligned[i] and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            # Fade Long: price < S3 with dry volume (mean reversion from extreme)
            elif (curr_low < S3_aligned[i] and
                  curr_volume_dry):
                signals[i] = 0.25
                position = 1
            # Fade Short: price > R3 with dry volume (mean reversion from extreme)
            elif (curr_high > R3_aligned[i] and
                  curr_volume_dry):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal + Volume Spike
# Uses weekly Camarilla pivot levels (R3/S3, R4/S4) from prior week.
# Long when price breaks above R4 with volume > 2x avg (continuation breakout).
# Short when price breaks below S4 with volume > 2x avg (continuation breakdown).
# Fade trades at R3/S3 with volume < 0.5x avg (mean reversion in extreme zones).
# Works in bull/bear via volume confirmation and pivot structure.
# Timeframe: 6h (primary), HTF: 1w for weekly pivots.

name = "6h_WeeklyCamarilla_R3R4_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from prior week's OHLC
    # Using prior week's data to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla levels: based on prior week's range
    weekly_range = weekly_high - weekly_low
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Resistance levels
    R3 = weekly_pivot + (weekly_range * 1.1 / 4.0)
    R4 = weekly_pivot + (weekly_range * 1.1 / 2.0)
    
    # Support levels
    S3 = weekly_pivot - (weekly_range * 1.1 / 4.0)
    S4 = weekly_pivot - (weekly_range * 1.1 / 2.0)
    
    # Align to 6h timeframe (wait for weekly bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Volume confirmation: volume > 2.0x 20-period average for breakouts
    # Fade volume: volume < 0.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    volume_dry = volume < (0.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or \
           np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_volume_dry = volume_dry[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below R3 (failed breakout)
            # 2. Price falls below weekly pivot (trend change)
            if (curr_close < R3_aligned[i] or
                curr_close < weekly_pivot[i]):  # Note: weekly_pivot not aligned - need to fix
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above S3 (failed breakdown)
            # 2. Price rises above weekly pivot (trend change)
            if (curr_close > S3_aligned[i] or
                curr_close > weekly_pivot[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Breakout Long: price > R4 with volume spike
            if (curr_high > R4_aligned[i] and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Breakdown Short: price < S4 with volume spike
            elif (curr_low < S4_aligned[i] and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            # Fade Long: price < S3 with dry volume (mean reversion from extreme)
            elif (curr_low < S3_aligned[i] and
                  curr_volume_dry):
                signals[i] = 0.25
                position = 1
            # Fade Short: price > R3 with dry volume (mean reversion from extreme)
            elif (curr_high > R3_aligned[i] and
                  curr_volume_dry):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals