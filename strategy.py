#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume Spike + 1d ADX Filter
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe
with volume confirmation and 1d ADX filter to trade in trending markets. Designed to work in both
bull and bear regimes by fading extremes in range and capturing breakouts in trends.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    r4 = pivot + (range_1d * 1.1)
    s4 = pivot - (range_1d * 1.1)
    
    # Calculate ADX on 1d for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    dm_plus_smooth = wilder_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilder_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period_adx)
    
    # Align Camarilla levels and ADX to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For pivot calculation stability
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below S3 OR ADX < 20
            if close[i] < s3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above R3 OR ADX < 20
            if close[i] > r3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Mean reversion: fade at R3/S3 when ADX < 25 (range market)
            # Breakout: break R4/S4 when ADX >= 25 (trending market)
            adx_val = adx_aligned[i]
            
            if adx_val < 25:  # Range market - mean reversion at S3/R3
                # Long when price touches S3 and shows rejection
                if close[i] <= s3_aligned[i] * 1.001 and close[i] > open[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short when price touches R3 and shows rejection
                elif close[i] >= r3_aligned[i] * 0.999 and close[i] < open[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
            else:  # Trending market - breakout at R4/S4
                # Long when price breaks R4 with volume
                if close[i] > r4_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks S4 with volume
                elif close[i] < s4_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>

#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume Spike + 1d ADX Filter
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe
with volume confirmation and 1d ADX filter to trade in trending markets. Designed to work in both
bull and bear regimes by fading extremes in range and capturing breakouts in trends.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    r4 = pivot + (range_1d * 1.1)
    s4 = pivot - (range_1d * 1.1)
    
    # Calculate ADX on 1d for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    dm_plus_smooth = wilder_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilder_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period_adx)
    
    # Align Camarilla levels and ADX to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For pivot calculation stability
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below S3 OR ADX < 20
            if close[i] < s3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above R3 OR ADX < 20
            if close[i] > r3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Mean reversion: fade at R3/S3 when ADX < 25 (range market)
            # Breakout: break R4/S4 when ADX >= 25 (trending market)
            adx_val = adx_aligned[i]
            
            if adx_val < 25:  # Range market - mean reversion at S3/R3
                # Long when price touches S3 and shows rejection
                if close[i] <= s3_aligned[i] * 1.001 and close[i] > open[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short when price touches R3 and shows rejection
                elif close[i] >= r3_aligned[i] * 0.999 and close[i] < open[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
            else:  # Trending market - breakout at R4/S4
                # Long when price breaks R4 with volume
                if close[i] > r4_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks S4 with volume
                elif close[i] < s4_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume Spike + 1d ADX Filter
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe
with volume confirmation and 1d ADX filter to trade in trending markets. Designed to work in both
bull and bear regimes by fading extremes in range and capturing breakouts in trends.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    r4 = pivot + (range_1d * 1.1)
    s4 = pivot - (range_1d * 1.1)
    
    # Calculate ADX on 1d for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    dm_plus_smooth = wilder_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilder_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period_adx)
    
    # Align Camarilla levels and ADX to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For pivot calculation stability
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below S3 OR ADX < 20
            if close[i] < s3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above R3 OR ADX < 20
            if close[i] > r3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Mean reversion: fade at R3/S3 when ADX < 25 (range market)
            # Breakout: break R4/S4 when ADX >= 25 (trending market)
            adx_val = adx_aligned[i]
            
            if adx_val < 25:  # Range market - mean reversion at S3/R3
                # Long when price touches S3 and shows rejection
                if close[i] <= s3_aligned[i] * 1.001 and close[i] > open[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short when price touches R3 and shows rejection
                elif close[i] >= r3_aligned[i] * 0.999 and close[i] < open[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
            else:  # Trending market - breakout at R4/S4
                # Long when price breaks R4 with volume
                if close[i] > r4_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks S4 with volume
                elif close[i] < s4_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume Spike + 1d ADX Filter
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe
with volume confirmation and 1d ADX filter to trade in trending markets. Designed to work in both
bull and bear regimes by fading extremes in range and capturing breakouts in trends.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    r4 = pivot + (range_1d * 1.1)
    s4 = pivot - (range_1d * 1.1)
    
    # Calculate ADX on 1d for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    dm_plus_smooth = wilder_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilder_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period_adx)
    
    # Align Camarilla levels and ADX to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For pivot calculation stability
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below S3 OR ADX < 20
            if close[i] < s3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above R3 OR ADX < 20
            if close[i] > r3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Mean reversion: fade at R3/S3 when ADX < 25 (range market)
            # Breakout: break R4/S4 when ADX >= 25 (trending market)
            adx_val = adx_aligned[i]
            
            if adx_val < 25:  # Range market - mean reversion at S3/R3
                # Long when price touches S3 and shows rejection
                if close[i] <= s3_aligned[i] * 1.001 and close[i] > open[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short when price touches R3 and shows rejection
                elif close[i] >= r3_aligned[i] * 0.999 and close[i] < open[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
            else:  # Trending market - breakout at R4/S4
                # Long when price breaks R4 with volume
                if close[i] > r4_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks S4 with volume
                elif close[i] < s4_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
    
    return signals

#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume Spike + 1d ADX Filter
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe
with volume confirmation and 1d ADX filter to trade in trending markets. Designed to work in both
bull and bear regimes by fading extremes in range and capturing breakouts in trends.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices