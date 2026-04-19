#!/usr/bin/env python3
"""
4h_WickReversal_SuperTrend_Filter
Hypothesis: 4-hour wick reversals (long lower wick for longs, long upper wick for shorts) 
combined with SuperTrend filter to trade with higher timeframe trend. 
Wick reversals signal exhaustion moves; SuperTrend filters for trend direction to 
avoid counter-trend trades. Works in bull/bear via trend filter.
Target: 20-50 trades/year via strict wick + trend requirements.
"""

name = "4h_WickReversal_SuperTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # SuperTrend calculation (ATR=10, multiplier=3.0)
    def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR using Wilder's smoothing
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                    else:
                        result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, atr_period)
        
        # Basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        # Final bands
        final_upper = np.full_like(close, np.nan)
        final_lower = np.full_like(close, np.nan)
        
        for i in range(len(close)):
            if i == 0:
                final_upper[i] = upper_band[i]
                final_lower[i] = lower_band[i]
            else:
                if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                    final_upper[i] = upper_band[i]
                else:
                    final_upper[i] = final_upper[i-1]
                
                if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                    final_lower[i] = lower_band[i]
                else:
                    final_lower[i] = final_lower[i-1]
        
        # SuperTrend
        supertrend = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i == 0:
                supertrend[i] = final_upper[i]
            else:
                if supertrend[i-1] == final_upper[i-1] and close[i] <= final_upper[i]:
                    supertrend[i] = final_upper[i]
                elif supertrend[i-1] == final_upper[i-1] and close[i] > final_upper[i]:
                    supertrend[i] = final_lower[i]
                elif supertrend[i-1] == final_lower[i-1] and close[i] >= final_lower[i]:
                    supertrend[i] = final_lower[i]
                elif supertrend[i-1] == final_lower[i-1] and close[i] < final_lower[i]:
                    supertrend[i] = final_upper[i]
        
        return supertrend
    
    # 4h data for SuperTrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    supertrend_4h = calculate_supertrend(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        10, 3.0
    )
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    
    # Wick calculations for reversal signals
    body_size = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    total_range = high - low
    
    # Avoid division by zero
    range_safe = np.where(total_range == 0, 1, total_range)
    upper_wick_ratio = upper_wick / range_safe
    lower_wick_ratio = lower_wick / range_safe
    
    # Strong wick rejection: wick > 60% of range and body < 40% of range
    strong_upper_wick = upper_wick_ratio > 0.6
    strong_lower_wick = lower_wick_ratio > 0.6
    small_body = body_size < (0.4 * range_safe)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if required data is NaN
        if (np.isnan(supertrend_4h_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            np.isnan(upper_wick_ratio[i]) or
            np.isnan(lower_wick_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from SuperTrend
        # SuperTrend acts as support in uptrend, resistance in downtrend
        is_uptrend = close[i] > supertrend_4h_aligned[i]
        
        if position == 0:
            # Long: strong lower wick rejection + uptrend + volume
            if (strong_lower_wick[i] and 
                small_body[i] and 
                is_uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: strong upper wick rejection + downtrend + volume
            elif (strong_upper_wick[i] and 
                  small_body[i] and 
                  not is_uptrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below SuperTrend (trend change) or opposite wick
            if (close[i] < supertrend_4h_aligned[i]) or (strong_upper_wick[i] and small_body[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above SuperTrend or opposite wick
            if (close[i] > supertrend_4h_aligned[i]) or (strong_lower_wick[i] and small_body[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Open variable fix
open_ = prices['open'].values if 'prices' in locals() else np.array([])  # This line will be removed/replaced
# Actually, let's properly define it:
# Re-defining the function with proper open variable handling

#!/usr/bin/env python3
"""
4h_WickReversal_SuperTrend_Filter
Hypothesis: 4-hour wick reversals (long lower wick for longs, long upper wick for shorts) 
combined with SuperTrend filter to trade with higher timeframe trend. 
Wick reversals signal exhaustion moves; SuperTrend filters for trend direction to 
avoid counter-trend trades. Works in bull/bear via trend filter.
Target: 20-50 trades/year via strict wick + trend requirements.
"""

name = "4h_WickReversal_SuperTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # SuperTrend calculation (ATR=10, multiplier=3.0)
    def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR using Wilder's smoothing
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                    else:
                        result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, atr_period)
        
        # Basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        # Final bands
        final_upper = np.full_like(close, np.nan)
        final_lower = np.full_like(close, np.nan)
        
        for i in range(len(close)):
            if i == 0:
                final_upper[i] = upper_band[i]
                final_lower[i] = lower_band[i]
            else:
                if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                    final_upper[i] = upper_band[i]
                else:
                    final_upper[i] = final_upper[i-1]
                
                if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                    final_lower[i] = lower_band[i]
                else:
                    final_lower[i] = final_lower[i-1]
        
        # SuperTrend
        supertrend = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i == 0:
                supertrend[i] = final_upper[i]
            else:
                if supertrend[i-1] == final_upper[i-1] and close[i] <= final_upper[i]:
                    supertrend[i] = final_upper[i]
                elif supertrend[i-1] == final_upper[i-1] and close[i] > final_upper[i]:
                    supertrend[i] = final_lower[i]
                elif supertrend[i-1] == final_lower[i-1] and close[i] >= final_lower[i]:
                    supertrend[i] = final_lower[i]
                elif supertrend[i-1] == final_lower[i-1] and close[i] < final_lower[i]:
                    supertrend[i] = final_upper[i]
        
        return supertrend
    
    # 4h data for SuperTrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    supertrend_4h = calculate_supertrend(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        10, 3.0
    )
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    
    # Wick calculations for reversal signals
    body_size = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    total_range = high - low
    
    # Avoid division by zero
    range_safe = np.where(total_range == 0, 1, total_range)
    upper_wick_ratio = upper_wick / range_safe
    lower_wick_ratio = lower_wick / range_safe
    
    # Strong wick rejection: wick > 60% of range and body < 40% of range
    strong_upper_wick = upper_wick_ratio > 0.6
    strong_lower_wick = lower_wick_ratio > 0.6
    small_body = body_size < (0.4 * range_safe)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if required data is NaN
        if (np.isnan(supertrend_4h_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            np.isnan(upper_wick_ratio[i]) or
            np.isnan(lower_wick_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from SuperTrend
        # SuperTrend acts as support in uptrend, resistance in downtrend
        is_uptrend = close[i] > supertrend_4h_aligned[i]
        
        if position == 0:
            # Long: strong lower wick rejection + uptrend + volume
            if (strong_lower_wick[i] and 
                small_body[i] and 
                is_uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: strong upper wick rejection + downtrend + volume
            elif (strong_upper_wick[i] and 
                  small_body[i] and 
                  not is_uptrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below SuperTrend (trend change) or opposite wick
            if (close[i] < supertrend_4h_aligned[i]) or (strong_upper_wick[i] and small_body[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above SuperTrend or opposite wick
            if (close[i] > supertrend_4h_aligned[i]) or (strong_lower_wick[i] and small_body[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals