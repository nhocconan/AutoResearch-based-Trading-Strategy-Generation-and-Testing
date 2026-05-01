#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation + ATR(14) stoploss.
# Long when price breaks above Donchian upper band AND 1d HMA uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below Donchian lower band AND 1d HMA downtrend AND volume > 1.5x 20-period median.
# Uses ATR-based trailing stop: exit long if price < highest_since_entry - 2.0*ATR, exit short if price > lowest_since_entry + 2.0*ATR.
# Discrete position sizing (0.25) minimizes fee churn. Target: 20-50 trades/year on 4h timeframe.
# Donchian provides objective structure, HMA filters noise with lag reduction, volume confirms breakout strength.
# This combination avoids overtrading while capturing trending moves in both bull and bear markets via symmetric long/short logic.

name = "4h_Donchian20_Breakout_1dHMA21_VolumeSpike_ATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HMA(21) for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d close
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        
        if len(wma_half) < half_period or len(wma_full) < period:
            return np.full_like(arr, np.nan)
        
        # Align arrays: wma_half starts at index half_period-1, wma_full at index period-1
        # We need to align them to the same index for subtraction
        diff = 2 * wma_half[half_period-1:] - wma_full[period-1:]
        # Pad diff to match original array length
        diff_padded = np.full_like(arr, np.nan)
        diff_padded[period-1:period-1+len(diff)] = diff
        
        # Final WMA of diff with sqrt_period
        hma = wma(diff_padded, sqrt_period)
        # HMA valid from index: period-1 + sqrt_period-1 onwards
        hma_valid_start = period-1 + sqrt_period-1
        if hma_valid_start < len(arr):
            hma[:hma_valid_start] = np.nan
            hma[hma_valid_start:] = hma[hma_valid_start:hma_valid_start+len(hma[hma_valid_start:])]
        return hma
    
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian(20) channels from 4h data
    # Upper band = 20-period high, Lower band = 20-period low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for HMA, ATR, Donchian, and volume median
    start_idx = max(21, 20, 14, 20)  # 21
    
    for i in range(start_idx, n):
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d HMA21 direction (using slope approximation)
        if i >= start_idx + 1:
            hma_now = hma_21_1d_aligned[i]
            hma_prev = hma_21_1d_aligned[i-1]
            uptrend = hma_now > hma_prev
            downtrend = hma_now < hma_prev
        else:
            uptrend = curr_close > hma_21_1d_aligned[i]
            downtrend = curr_close < hma_21_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_upper[i-1]  # Using previous bar's upper band
        breakout_down = curr_low < donchian_lower[i-1]  # Using previous bar's lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume spike
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Breakout down AND downtrend AND volume spike
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = highest_since_entry - 2.0 * curr_atr
            if curr_close < stop_price or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stop_price or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation + ATR(14) stoploss.
# Long when price breaks above Donchian upper band AND 1d HMA uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below Donchian lower band AND 1d HMA downtrend AND volume > 1.5x 20-period median.
# Uses ATR-based trailing stop: exit long if price < highest_since_entry - 2.0*ATR, exit short if price > lowest_since_entry + 2.0*ATR.
# Discrete position sizing (0.25) minimizes fee churn. Target: 20-50 trades/year on 4h timeframe.
# Donchian provides objective structure, HMA filters noise with lag reduction, volume confirms breakout strength.
# This combination avoids overtrading while capturing trending moves in both bull and bear markets via symmetric long/short logic.

name = "4h_Donchian20_Breakout_1dHMA21_VolumeSpike_ATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HMA(21) for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d close
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        
        if len(wma_half) < half_period or len(wma_full) < period:
            return np.full_like(arr, np.nan)
        
        # Align arrays: wma_half starts at index half_period-1, wma_full at index period-1
        # We need to align them to the same index for subtraction
        diff = 2 * wma_half[half_period-1:] - wma_full[period-1:]
        # Pad diff to match original array length
        diff_padded = np.full_like(arr, np.nan)
        diff_padded[period-1:period-1+len(diff)] = diff
        
        # Final WMA of diff with sqrt_period
        hma = wma(diff_padded, sqrt_period)
        # HMA valid from index: period-1 + sqrt_period-1 onwards
        hma_valid_start = period-1 + sqrt_period-1
        if hma_valid_start < len(arr):
            hma[:hma_valid_start] = np.nan
            hma[hma_valid_start:] = hma[hma_valid_start:hma_valid_start+len(hma[hma_valid_start:])]
        return hma
    
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian(20) channels from 4h data
    # Upper band = 20-period high, Lower band = 20-period low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for HMA, ATR, Donchian, and volume median
    start_idx = max(21, 20, 14, 20)  # 21
    
    for i in range(start_idx, n):
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d HMA21 direction (using slope approximation)
        if i >= start_idx + 1:
            hma_now = hma_21_1d_aligned[i]
            hma_prev = hma_21_1d_aligned[i-1]
            uptrend = hma_now > hma_prev
            downtrend = hma_now < hma_prev
        else:
            uptrend = curr_close > hma_21_1d_aligned[i]
            downtrend = curr_close < hma_21_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_upper[i-1]  # Using previous bar's upper band
        breakout_down = curr_low < donchian_lower[i-1]  # Using previous bar's lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume spike
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Breakout down AND downtrend AND volume spike
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = highest_since_entry - 2.0 * curr_atr
            if curr_close < stop_price or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stop_price or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation + ATR(14) stoploss.
# Long when price breaks above Donchian upper band AND 1d HMA uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below Donchian lower band AND 1d HMA downtrend AND volume > 1.5x 20-period median.
# Uses ATR-based trailing stop: exit long if price < highest_since_entry - 2.0*ATR, exit short if price > lowest_since_entry + 2.0*ATR.
# Discrete position sizing (0.25) minimizes fee churn. Target: 20-50 trades/year on 4h timeframe.
# Donchian provides objective structure, HMA filters noise with lag reduction, volume confirms breakout strength.
# This combination avoids overtrading while capturing trending moves in both bull and bear markets via symmetric long/short logic.

name = "4h_Donchian20_Breakout_1dHMA21_VolumeSpike_ATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HMA(21) for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d close
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        
        if len(wma_half) < half_period or len(wma_full) < period:
            return np.full_like(arr, np.nan)
        
        # Align arrays: wma_half starts at index half_period-1, wma_full at index period-1
        # We need to align them to the same index for subtraction
        diff = 2 * wma_half[half_period-1:] - wma_full[period-1:]
        # Pad diff to match original array length
        diff_padded = np.full_like(arr, np.nan)
        diff_padded[period-1:period-1+len(diff)] = diff
        
        # Final WMA of diff with sqrt_period
        hma = wma(diff_padded, sqrt_period)
        # HMA valid from index: period-1 + sqrt_period-1 onwards
        hma_valid_start = period-1 + sqrt_period-1
        if hma_valid_start < len(arr):
            hma[:hma_valid_start] = np.nan
            hma[hma_valid_start:] = hma[hma_valid_start:hma_valid_start+len(hma[hma_valid_start:])]
        return hma
    
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian(20) channels from 4h data
    # Upper band = 20-period high, Lower band = 20-period low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for HMA, ATR, Donchian, and volume median
    start_idx = max(21, 20, 14, 20)  # 21
    
    for i in range(start_idx, n):
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d HMA21 direction (using slope approximation)
        if i >= start_idx + 1:
            hma_now = hma_21_1d_aligned[i]
            hma_prev = hma_21_1d_aligned[i-1]
            uptrend = hma_now > hma_prev
            downtrend = hma_now < hma_prev
        else:
            uptrend = curr_close > hma_21_1d_aligned[i]
            downtrend = curr_close < hma_21_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_upper[i-1]  # Using previous bar's upper band
        breakout_down = curr_low < donchian_lower[i-1]  # Using previous bar's lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume spike
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Breakout down AND downtrend AND volume spike
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = highest_since_entry - 2.0 * curr_atr
            if curr_close < stop_price or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stop_price or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation + ATR(14) stoploss.
# Long when price breaks above Donchian upper band AND 1d HMA uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below Donchian lower band AND 1d HMA downtrend AND volume > 1.5x 20-period median.
# Uses ATR-based trailing stop: exit long if price < highest_since_entry - 2.0*ATR, exit short if price > lowest_since_entry + 2.0*ATR.
# Discrete position sizing (0.25) minimizes fee churn. Target: 20-50 trades/year on 4h timeframe.
# Donchian provides objective structure, HMA filters noise with lag reduction, volume confirms breakout strength.
# This combination avoids overtrading while capturing trending moves in both bull and bear markets via symmetric long/short logic.

name = "4h_Donchian20_Breakout_1dHMA21_VolumeSpike_ATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HMA(21) for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d close
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        
        if len(wma_half) < half_period or len(wma_full) < period:
            return np.full_like(arr, np.nan)
        
        # Align arrays: wma_half starts at index half_period-1, wma_full at index period-1
        # We need to align them to the same index for subtraction
        diff = 2 * wma_half[half_period-1:] - wma_full[period-1:]
        # Pad diff to match original array length
        diff_padded = np.full_like(arr, np.nan)
        diff_padded[period-1:period-1+len(diff)] = diff
        
        # Final WMA of diff with sqrt_period
        hma = wma(diff_padded, sqrt_period)
        # HMA valid from index: period-1 + sqrt_period-1 onwards
        hma_valid_start = period-1 + sqrt_period-1
        if hma_valid_start < len(arr):
            hma[:hma_valid_start] = np.nan
            hma[hma_valid_start:] = hma[hma_valid_start:hma_valid_start+len(hma[hma_valid_start:])]
        return hma
    
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian(20) channels from 4h data
    # Upper band = 20-period high, Lower band = 20-period low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for HMA, ATR, Donchian, and volume median
    start_idx = max(21, 20, 14, 20)  # 21
    
    for i in range(start_idx, n):
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d HMA21 direction (using slope approximation)
        if i >= start_idx + 1:
            hma_now = hma_21_1d_aligned[i]
            hma_prev = hma_21_1d_aligned[i-1]
            uptrend = hma_now > hma_prev
            downtrend = hma_now < hma_prev
        else:
            uptrend = curr_close > hma_21_1d_aligned[i]
            downtrend = curr_close < hma_21_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_upper[i-1]  # Using previous bar's upper band
        breakout_down = curr_low < donchian_lower[i-1]  # Using previous bar's lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume spike
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Breakout down AND downtrend AND volume