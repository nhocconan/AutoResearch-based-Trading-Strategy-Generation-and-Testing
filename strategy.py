#!/usr/bin/env python3
"""
exp_7347_6h_donchian20_1w_pivot_dir_v1
Hypothesis: 6h Donchian(20) breakout with 1d/1w pivot direction filter and volume confirmation.
Uses 1d for pivot levels and 1w for trend regime to reduce noise. Targets 75-150 trades over 4 years.
Discrete position sizing (0.0, ±0.25) minimizes fee churn. Works in bull/bear via weekly pivot trend filter.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7347_6h_donchian20_1w_pivot_dir_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 10  # bars for swing high/low
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivots and 1w for trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d pivot points (using previous day's H/L/C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    r1 = 2 * pivot - low_1d[:-1]
    s1 = 2 * pivot - high_1d[:-1]
    r2 = pivot + (high_1d[:-1] - low_1d[:-1])
    s2 = pivot - (high_1d[:-1] - low_1d[:-1])
    r3 = high_1d[:-1] + 2 * (pivot - low_1d[:-1])
    s3 = low_1d[:-1] - 2 * (high_1d[:-1] - pivot)
    
    # Shift to align with current day (use previous day's pivots)
    pivot = np.concatenate([ [np.nan], pivot ])
    r1 = np.concatenate([ [np.nan], r1 ])
    s1 = np.concatenate([ [np.nan], s1 ])
    r2 = np.concatenate([ [np.nan], r2 ])
    s2 = np.concatenate([ [np.nan], s2 ])
    r3 = np.concatenate([ [np.nan], r3 ])
    s3 = np.concatenate([ [np.nan], s3 ])
    
    # Align to LTF (6h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1w EMA for trend regime
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD, 50) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend from weekly EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        # Use daily pivots for entry/exit levels
        # Long conditions: price above weekly EMA (bullish regime) + break above R1 with volume
        long_breakout = above_ema and (close[i] > r1_aligned[i]) and vol_confirmed
        # Long retest: pullback to pivot or S1 with volume in bullish regime
        long_retest = above_ema and ((abs(close[i] - pivot_aligned[i]) < pivot_aligned[i] * 0.002) or 
                                     (abs(close[i] - s1_aligned[i]) < s1_aligned[i] * 0.002)) and vol_confirmed
        
        # Short conditions: price below weekly EMA (bearish regime) + break below S1 with volume
        short_breakout = below_ema and (close[i] < s1_aligned[i]) and vol_confirmed
        # Short retest: pullback to pivot or R1 with volume in bearish regime
        short_retest = below_ema and ((abs(close[i] - pivot_aligned[i]) < pivot_aligned[i] * 0.002) or 
                                      (abs(close[i] - r1_aligned[i]) < r1_aligned[i] * 0.002)) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout or long_retest:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout or short_retest:
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
exp_7347_6h_donchian20_1w_pivot_dir_v1
Hypothesis: 6h Donchian(20) breakout with 1d/1w pivot direction filter and volume confirmation.
Uses 1d for pivot levels and 1w for trend regime to reduce noise. Targets 75-150 trades over 4 years.
Discrete position sizing (0.0, ±0.25) minimizes fee churn. Works in bull/bear via weekly pivot trend filter.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7347_6h_donchian20_1w_pivot_dir_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 10  # bars for swing high/low
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivots and 1w for trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d pivot points (using previous day's H/L/C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    r1 = 2 * pivot - low_1d[:-1]
    s1 = 2 * pivot - high_1d[:-1]
    r2 = pivot + (high_1d[:-1] - low_1d[:-1])
    s2 = pivot - (high_1d[:-1] - low_1d[:-1])
    r3 = high_1d[:-1] + 2 * (pivot - low_1d[:-1])
    s3 = low_1d[:-1] - 2 * (high_1d[:-1] - pivot)
    
    # Shift to align with current day (use previous day's pivots)
    pivot = np.concatenate([ [np.nan], pivot ])
    r1 = np.concatenate([ [np.nan], r1 ])
    s1 = np.concatenate([ [np.nan], s1 ])
    r2 = np.concatenate([ [np.nan], r2 ])
    s2 = np.concatenate([ [np.nan], s2 ])
    r3 = np.concatenate([ [np.nan], r3 ])
    s3 = np.concatenate([ [np.nan], s3 ])
    
    # Align to LTF (6h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1w EMA for trend regime
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD, 50) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend from weekly EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        # Use daily pivots for entry/exit levels
        # Long conditions: price above weekly EMA (bullish regime) + break above R1 with volume
        long_breakout = above_ema and (close[i] > r1_aligned[i]) and vol_confirmed
        # Long retest: pullback to pivot or S1 with volume in bullish regime
        long_retest = above_ema and ((abs(close[i] - pivot_aligned[i]) < pivot_aligned[i] * 0.002) or 
                                     (abs(close[i] - s1_aligned[i]) < s1_aligned[i] * 0.002)) and vol_confirmed
        
        # Short conditions: price below weekly EMA (bearish regime) + break below S1 with volume
        short_breakout = below_ema and (close[i] < s1_aligned[i]) and vol_confirmed
        # Short retest: pullback to pivot or R1 with volume in bearish regime
        short_retest = below_ema and ((abs(close[i] - pivot_aligned[i]) < pivot_aligned[i] * 0.002) or 
                                      (abs(close[i] - r1_aligned[i]) < r1_aligned[i] * 0.002)) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout or long_retest:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout or short_retest:
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