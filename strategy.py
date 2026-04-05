#!/usr/bin/env python3
"""
exp_6991_6h_camarilla1d_pivot_v1
Hypothesis: 6h Camarilla pivot levels from 1d timeframe with volume confirmation.
In ranging markets (ADX < 25): fade at R3/S3 levels. In trending markets (ADX >= 25): 
breakout continuation at R4/S4 levels. Uses 1d HTF for pivot calculation and regime detection.
Designed for 6h timeframe to capture ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to volatility regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6991_6h_camarilla1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous 1d bar for Camarilla calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 20  # ~5 days (6h bars)
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    pivot_range = high_1d - low_1d
    r4_1d = close_1d + 1.5 * pivot_range
    r3_1d = close_1d + 1.0 * pivot_range
    s3_1d = close_1d - 1.0 * pivot_range
    s4_1d = close_1d - 1.5 * pivot_range
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed 1d bar only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d ADX for regime detection
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM = max(high - previous_high, 0) if > max(previous_low - low, 0) else 0
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    # -DM = max(previous_low - low, 0) if > max(high - previous_high, 0) else 0
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smooth TR, +DM, -DM
    atr_1d = tr_1d.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_plus_smooth = dm_plus.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_minus_smooth = dm_minus.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = dx.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss (6h)
    tr1_ltf = pd.Series(high - low)
    tr2_ltf = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3_ltf = pd.Series(np.abs(low - np.roll(close, 1)))
    tr_ltf = pd.concat([tr1_ltf, tr2_ltf, tr3_ltf], axis=1).max(axis=1)
    atr_ltf = tr_ltf.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK + ADX_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr_ltf[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr_ltf[i]:
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
        
        # Determine market regime from 1d ADX
        is_ranging = adx_1d_aligned[i] < ADX_TREND_THRESHOLD
        is_trending = adx_1d_aligned[i] >= ADX_TREND_THRESHOLD
        
        # Camarilla-based signals
        if is_ranging:
            # In ranging markets: fade at R3/S3
            long_signal = (close[i] <= s3_1d_aligned[i]) and vol_confirmed
            short_signal = (close[i] >= r3_1d_aligned[i]) and vol_confirmed
        else:
            # In trending markets: breakout at R4/S4
            long_signal = (close[i] >= r4_1d_aligned[i]) and vol_confirmed
            short_signal = (close[i] <= s4_1d_aligned[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
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
exp_6991_6h_camarilla1d_pivot_v1
Hypothesis: 6h Camarilla pivot levels from 1d timeframe with volume confirmation.
In ranging markets (ADX < 25): fade at R3/S3 levels. In trending markets (ADX >= 25): 
breakout continuation at R4/S4 levels. Uses 1d HTF for pivot calculation and regime detection.
Designed for 6h timeframe to capture ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to volatility regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6991_6h_camarilla1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous 1d bar for Camarilla calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 20  # ~5 days (6h bars)
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    pivot_range = high_1d - low_1d
    r4_1d = close_1d + 1.5 * pivot_range
    r3_1d = close_1d + 1.0 * pivot_range
    s3_1d = close_1d - 1.0 * pivot_range
    s4_1d = close_1d - 1.5 * pivot_range
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed 1d bar only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d ADX for regime detection
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM = max(high - previous_high, 0) if > max(previous_low - low, 0) else 0
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    # -DM = max(previous_low - low, 0) if > max(high - previous_high, 0) else 0
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smooth TR, +DM, -DM
    atr_1d = tr_1d.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_plus_smooth = dm_plus.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_minus_smooth = dm_minus.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = dx.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss (6h)
    tr1_ltf = pd.Series(high - low)
    tr2_ltf = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3_ltf = pd.Series(np.abs(low - np.roll(close, 1)))
    tr_ltf = pd.concat([tr1_ltf, tr2_ltf, tr3_ltf], axis=1).max(axis=1)
    atr_ltf = tr_ltf.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK + ADX_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr_ltf[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr_ltf[i]:
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
        
        # Determine market regime from 1d ADX
        is_ranging = adx_1d_aligned[i] < ADX_TREND_THRESHOLD
        is_trending = adx_1d_aligned[i] >= ADX_TREND_THRESHOLD
        
        # Camarilla-based signals
        if is_ranging:
            # In ranging markets: fade at R3/S3
            long_signal = (close[i] <= s3_1d_aligned[i]) and vol_confirmed
            short_signal = (close[i] >= r3_1d_aligned[i]) and vol_confirmed
        else:
            # In trending markets: breakout at R4/S4
            long_signal = (close[i] >= r4_1d_aligned[i]) and vol_confirmed
            short_signal = (close[i] <= s4_1d_aligned[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
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