#!/usr/bin/env python3
"""
Experiment #8811: 6h Donchian(20) breakout + 1d pivot direction + volume confirmation
Hypothesis: 6h timeframe balances trade frequency (target 50-150/4y) with responsiveness.
Using 1-day Camarilla pivot levels for directional bias: price above pivot = bullish bias,
price below pivot = bearish bias. Donchian breakout with volume confirmation ensures
institutional participation. Works in bull (breakouts) and bear (fades at extreme levels).
Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
"""

from mtf_data import get_ftf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8811_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous day's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels"""
    # Camarilla formulas
    pivot = (high + low + close) / 3
    range_val = high - low
    
    # Resistance levels
    r1 = close + range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    
    # Support levels
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for pivot levels
    pivot = np.full(len(close_1d), np.nan)
    r1 = np.full(len(close_1d), np.nan)
    r2 = np.full(len(close_1d), np.nan)
    r3 = np.full(len(close_1d), np.nan)
    r4 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    s2 = np.full(len(close_1d), np.nan)
    s3 = np.full(len(close_1d), np.nan)
    s4 = np.full(len(close_1d), np.nan)
    
    # Calculate pivot for each day (starting from PIVOT_LOOKBACK)
    for i in range(PIVOT_LOOKBACK, len(close_1d)):
        pivot[i], r1[i], r2[i], r3[i], r4[i], s1[i], s2[i], s3[i], s4[i] = \
            calculate_camarilla_pivot(high_1d[i-PIVOT_LOOKBACK], low_1d[i-PIVOT_LOOKBACK], close_1d[i-PIVOT_LOOKBACK])
    
    # Determine bias: price above pivot = bullish, below = bearish
    # Using R3/S3 as extreme levels for fade signals
    bull_bias = close_1d > pivot  # Price above daily pivot
    bear_bias = close_1d < pivot  # Price below daily pivot
    extreme_long = close_1d <= s3  # At or below S3 - extreme oversold
    extreme_short = close_1d >= r3  # At or above R3 - extreme overbought
    
    # Align to 6h timeframe
    bull_bias_aligned = align_htf_to_ltf(prices, df_1d, bull_bias.astype(float))
    bear_bias_aligned = align_htf_to_ltf(prices, df_1d, bear_bias.astype(float))
    extreme_long_aligned = align_htf_to_ltf(prices, df_1d, extreme_long.astype(float))
    extreme_short_aligned = align_htf_to_ltf(prices, df_1d, extreme_short.astype(float))
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(bull_bias_aligned[i]) or np.isnan(bear_bias_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d pivot
        bull_bias = bull_bias_aligned[i] == 1.0   # Price above daily pivot
        bear_bias = bear_bias_aligned[i] == 1.0   # Price below daily pivot
        extreme_long = extreme_long_aligned[i] == 1.0  # At/S below S3
        extreme_short = extreme_short_aligned[i] == 1.0  # At/Above R3
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions:
        # 1. Trend-following: breakout in direction of bias
        # 2. Mean-reversion: fade at extreme levels (R3/S3)
        long_entry = (bull_bias and long_breakout and volume_confirmed) or \
                     (extreme_long and long_breakout and volume_confirmed)
        short_entry = (bear_bias and short_breakout and volume_confirmed) or \
                      (extreme_short and short_breakout and volume_confirmed)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8811: 6h Donchian(20) breakout + 1d pivot direction + volume confirmation
Hypothesis: 6h timeframe balances trade frequency (target 50-150/4y) with responsiveness.
Using 1-day Camarilla pivot levels for directional bias: price above pivot = bullish bias,
price below pivot = bearish bias. Donchian breakout with volume confirmation ensures
institutional participation. Works in bull (breakouts) and bear (fades at extreme levels).
Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8811_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous day's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels"""
    # Camarilla formulas
    pivot = (high + low + close) / 3
    range_val = high - low
    
    # Resistance levels
    r1 = close + range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    
    # Support levels
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for pivot levels
    pivot = np.full(len(close_1d), np.nan)
    r1 = np.full(len(close_1d), np.nan)
    r2 = np.full(len(close_1d), np.nan)
    r3 = np.full(len(close_1d), np.nan)
    r4 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    s2 = np.full(len(close_1d), np.nan)
    s3 = np.full(len(close_1d), np.nan)
    s4 = np.full(len(close_1d), np.nan)
    
    # Calculate pivot for each day (starting from PIVOT_LOOKBACK)
    for i in range(PIVOT_LOOKBACK, len(close_1d)):
        pivot[i], r1[i], r2[i], r3[i], r4[i], s1[i], s2[i], s3[i], s4[i] = \
            calculate_camarilla_pivot(high_1d[i-PIVOT_LOOKBACK], low_1d[i-PIVOT_LOOKBACK], close_1d[i-PIVOT_LOOKBACK])
    
    # Determine bias: price above pivot = bullish, below = bearish
    # Using R3/S3 as extreme levels for fade signals
    bull_bias = close_1d > pivot  # Price above daily pivot
    bear_bias = close_1d < pivot  # Price below daily pivot
    extreme_long = close_1d <= s3  # At or below S3 - extreme oversold
    extreme_short = close_1d >= r3  # At or above R3 - extreme overbought
    
    # Align to 6h timeframe
    bull_bias_aligned = align_htf_to_ltf(prices, df_1d, bull_bias.astype(float))
    bear_bias_aligned = align_htf_to_ltf(prices, df_1d, bear_bias.astype(float))
    extreme_long_aligned = align_htf_to_ltf(prices, df_1d, extreme_long.astype(float))
    extreme_short_aligned = align_htf_to_ltf(prices, df_1d, extreme_short.astype(float))
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(bull_bias_aligned[i]) or np.isnan(bear_bias_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d pivot
        bull_bias = bull_bias_aligned[i] == 1.0   # Price above daily pivot
        bear_bias = bear_bias_aligned[i] == 1.0   # Price below daily pivot
        extreme_long = extreme_long_aligned[i] == 1.0  # At/S below S3
        extreme_short = extreme_short_aligned[i] == 1.0  # At/Above R3
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions:
        # 1. Trend-following: breakout in direction of bias
        # 2. Mean-reversion: fade at extreme levels (R3/S3)
        long_entry = (bull_bias and long_breakout and volume_confirmed) or \
                     (extreme_long and long_breakout and volume_confirmed)
        short_entry = (bear_bias and short_breakout and volume_confirmed) or \
                      (extreme_short and short_breakout and volume_confirmed)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals
</x>