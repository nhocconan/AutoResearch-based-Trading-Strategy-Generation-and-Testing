#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour donchian channel breakout with daily pivot point reversal and weekly volume confirmation.
# Donchian breakouts capture trend continuation, while pivot reversals (at S1/R1) exploit mean reversion in ranging markets.
# Weekly volume ensures institutional participation. Designed for 6h to target 50-150 trades over 4 years.

name = "6h_donchian20_1d_pivot1w_vol_v1"
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
    
    # 1-day Pivot Points (classic)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(5, len(vol_1w)):  # 6-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-5:i+1])
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # 6h Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 6)  # Donchian needs 19, volume needs 5
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.2x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.2
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches R1 (pivot resistance) or stoploss
            if (close[i] >= r1_aligned[i] or 
                close[i] < entry_price - 2.5 * (donchian_high[i] - donchian_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches S1 (pivot support) or stoploss
            if (close[i] <= s1_aligned[i] or 
                close[i] > entry_price + 2.5 * (donchian_high[i] - donchian_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: breakout above Donchian high with pivot support
                if (close[i] > donchian_high[i] and 
                    close[i] <= s1_aligned[i] * 1.02):  # near S1 support
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low with pivot resistance
                elif (close[i] < donchian_low[i] and 
                      close[i] >= r1_aligned[i] * 0.98):  # near R1 resistance
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                # Long: bounce from S1 support in ranging market
                elif (close[i] <= s1_aligned[i] * 1.01 and 
                      close[i] > close[i-1] and 
                      donchian_high[i] - donchian_low[i] < (close[i] * 0.05)):  # low volatility
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bounce from R1 resistance in ranging market
                elif (close[i] >= r1_aligned[i] * 0.99 and 
                      close[i] < close[i-1] and 
                      donchian_high[i] - donchian_low[i] < (close[i] * 0.05)):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour donchian channel breakout with daily pivot point reversal and weekly volume confirmation.
# Donchian breakouts capture trend continuation, while pivot reversals (at S1/R1) exploit mean reversion in ranging markets.
# Weekly volume ensures institutional participation. Designed for 6h to target 50-150 trades over 4 years.

name = "6h_donchian20_1d_pivot1w_vol_v1"
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
    
    # 1-day Pivot Points (classic)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(5, len(vol_1w)):  # 6-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-5:i+1])
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # 6h Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 6)  # Donchian needs 19, volume needs 5
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.2x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.2
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches R1 (pivot resistance) or stoploss
            if (close[i] >= r1_aligned[i] or 
                close[i] < entry_price - 2.5 * (donchian_high[i] - donchian_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches S1 (pivot support) or stoploss
            if (close[i] <= s1_aligned[i] or 
                close[i] > entry_price + 2.5 * (donchian_high[i] - donchian_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: breakout above Donchian high with pivot support
                if (close[i] > donchian_high[i] and 
                    close[i] <= s1_aligned[i] * 1.02):  # near S1 support
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low with pivot resistance
                elif (close[i] < donchian_low[i] and 
                      close[i] >= r1_aligned[i] * 0.98):  # near R1 resistance
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                # Long: bounce from S1 support in ranging market
                elif (close[i] <= s1_aligned[i] * 1.01 and 
                      close[i] > close[i-1] and 
                      donchian_high[i] - donchian_low[i] < (close[i] * 0.05)):  # low volatility
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bounce from R1 resistance in ranging market
                elif (close[i] >= r1_aligned[i] * 0.99 and 
                      close[i] < close[i-1] and 
                      donchian_high[i] - donchian_low[i] < (close[i] * 0.05)):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour donchian channel breakout with daily pivot point reversal and weekly volume confirmation.
# Donchian breakouts capture trend continuation, while pivot reversals (at S1/R1) exploit mean reversion in ranging markets.
# Weekly volume ensures institutional participation. Designed for 6h to target 50-150 trades over 4 years.

name = "6h_donchian20_1d_pivot1w_vol_v1"
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
    
    # 1-day Pivot Points (classic)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(5, len(vol_1w)):  # 6-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-5:i+1])
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # 6h Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 6)  # Donchian needs 19, volume needs 5
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.2x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.2
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches R1 (pivot resistance) or stoploss
            if (close[i] >= r1_aligned[i] or 
                close[i] < entry_price - 2.5 * (donchian_high[i] - donchian_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches S1 (pivot support) or stoploss
            if (close[i] <= s1_aligned[i] or 
                close[i] > entry_price + 2.5 * (donchian_high[i] - donchian_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: breakout above Donchian high with pivot support
                if (close[i] > donchian_high[i] and 
                    close[i] <= s1_aligned[i] * 1.02):  # near S1 support
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low with pivot resistance
                elif (close[i] < donchian_low[i] and 
                      close[i] >= r1_aligned[i] * 0.98):  # near R1 resistance
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                # Long: bounce from S1 support in ranging market
                elif (close[i] <= s1_aligned[i] * 1.01 and 
                      close[i] > close[i-1] and 
                      donchian_high[i] - donchian_low[i] < (close[i] * 0.05)):  # low volatility
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bounce from R1 resistance in ranging market
                elif (close[i] >= r1_aligned[i] * 0.99 and 
                      close[i] < close[i-1] and 
                      donchian_high[i] - donchian_low[i] < (close[i] * 0.05)):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals