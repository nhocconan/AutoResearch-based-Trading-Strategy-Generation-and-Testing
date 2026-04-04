#!/usr/bin/env python3
"""
exp_6568_12h_donchian20_1w_hma_vol_v1
Hypothesis: 12h Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation.
Long when price breaks above upper Donchian channel with HMA up and volume spike.
Short when price breaks below lower Donchian channel with HMA down and volume spike.
Exit on opposite Donchian band touch or max hold. Works in bull/bear via trend filter.
Target: 50-150 trades over 4 years (12-37/year). HARD MAX: 200 total.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6568_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25      # 25% position size
MAX_HOLD_BARS = 20      # max 20*12h = 10 days
HMA_PERIOD = 21

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for HMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA(21)
    close_1w = df_1w['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function
    def wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_1w, half_period)
    wma_full = wma(close_1w, HMA_PERIOD)
    # Pad to same length
    wma_half_padded = np.full_like(close_1w, np.nan)
    wma_half_padded[half_period-1:] = wma_half
    wma_full_padded = np.full_like(close_1w, np.nan)
    wma_full_padded[HMA_PERIOD-1:] = wma_full
    hull_raw = 2 * wma_half_padded - wma_full_padded
    hma_1w = wma(hull_raw, sqrt_period)
    # Pad HMA
    hma_1w_padded = np.full_like(close_1w, np.nan)
    hma_1w_padded[sqrt_period-1:] = hma_1w
    
    # Align HMA to LTF (12h) with shift(1)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_padded)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    upper_donchian = rolling_max(high, DONCHIAN_PERIOD)
    lower_donchian = rolling_min(low, DONCHIAN_PERIOD)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, HMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Determine trend from HMA slope (using 3-bar slope)
        if i >= 3:
            hma_slope = hma_1w_aligned[i] - hma_1w_aligned[i-3]
            hma_up = hma_slope > 0
            hma_down = hma_slope < 0
        else:
            hma_up = False
            hma_down = False
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions
        if position == 1:  # long position
            # Exit if price touches lower Donchian OR time-based exit
            exit_long = close[i] <= lower_donchian[i]
            exit_long = exit_long or bars_since_entry >= MAX_HOLD_BARS
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            # Exit if price touches upper Donchian OR time-based exit
            exit_short = close[i] >= upper_donchian[i]
            exit_short = exit_short or bars_since_entry >= MAX_HOLD_BARS
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            # Breakout long: price > upper Donchian with HMA up and volume confirmation
            if close[i] > upper_donchian[i] and hma_up and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Breakout short: price < lower Donchian with HMA down and volume confirmation
            elif close[i] < lower_donchian[i] and hma_down and vol_confirmed:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_6568_12h_donchian20_1w_hma_vol_v1
Hypothesis: 12h Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation.
Long when price breaks above upper Donchian channel with HMA up and volume spike.
Short when price breaks below lower Donchian channel with HMA down and volume spike.
Exit on opposite Donchian band touch or max hold. Works in bull/bear via trend filter.
Target: 50-150 trades over 4 years (12-37/year). HARD MAX: 200 total.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6568_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25      # 25% position size
MAX_HOLD_BARS = 20      # max 20*12h = 10 days
HMA_PERIOD = 21

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for HMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA(21)
    close_1w = df_1w['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function
    def wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_1w, half_period)
    wma_full = wma(close_1w, HMA_PERIOD)
    # Pad to same length
    wma_half_padded = np.full_like(close_1w, np.nan)
    wma_half_padded[half_period-1:] = wma_half
    wma_full_padded = np.full_like(close_1w, np.nan)
    wma_full_padded[HMA_PERIOD-1:] = wma_full
    hull_raw = 2 * wma_half_padded - wma_full_padded
    hma_1w = wma(hull_raw, sqrt_period)
    # Pad HMA
    hma_1w_padded = np.full_like(close_1w, np.nan)
    hma_1w_padded[sqrt_period-1:] = hma_1w
    
    # Align HMA to LTF (12h) with shift(1)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_padded)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    upper_donchian = rolling_max(high, DONCHIAN_PERIOD)
    lower_donchian = rolling_min(low, DONCHIAN_PERIOD)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, HMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Determine trend from HMA slope (using 3-bar slope)
        if i >= 3:
            hma_slope = hma_1w_aligned[i] - hma_1w_aligned[i-3]
            hma_up = hma_slope > 0
            hma_down = hma_slope < 0
        else:
            hma_up = False
            hma_down = False
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions
        if position == 1:  # long position
            # Exit if price touches lower Donchian OR time-based exit
            exit_long = close[i] <= lower_donchian[i]
            exit_long = exit_long or bars_since_entry >= MAX_HOLD_BARS
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            # Exit if price touches upper Donchian OR time-based exit
            exit_short = close[i] >= upper_donchian[i]
            exit_short = exit_short or bars_since_entry >= MAX_HOLD_BARS
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            # Breakout long: price > upper Donchian with HMA up and volume confirmation
            if close[i] > upper_donchian[i] and hma_up and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Breakout short: price < lower Donchian with HMA down and volume confirmation
            elif close[i] < lower_donchian[i] and hma_down and vol_confirmed:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals