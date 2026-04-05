#!/usr/bin/env python3
"""
exp_6971_6h_camarilla1d_pivot_vol_v1
Hypothesis: 6h Camarilla pivot breakout/fade with 1d volume confirmation.
In ranging markets (ADX < 25): fade at R3/S3 levels. In trending markets (ADX >= 25): breakout continuation at R4/S4.
Uses 1d Camarilla pivots calculated from prior 1d OHLC. ADX filter prevents whipsaw in low volatility.
Designed for 6h timeframe to capture ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to regime via ADX.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6971_6h_camarilla1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use prior 1d for Camarilla calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
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
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for prior 1d
    # Camarilla: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, etc.
    rng = high_1d - low_1d
    camarilla_r4 = close_1d + rng * 1.1 / 2
    camarilla_r3 = close_1d + rng * 1.1 / 4
    camarilla_s3 = close_1d - rng * 1.1 / 4
    camarilla_s4 = close_1d - rng * 1.1 / 2
    
    # Align Camarilla levels to 6h (shifted by 1 for completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d ADX for regime filter
    # TR calculation
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM and -DM
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1d = tr_1d.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    plus_dm_1d = pd.Series(up_move).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    minus_dm_1d = pd.Series(down_move).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # DI and ADX
    plus_di_1d = 100 * plus_dm_1d / (atr_1d + 1e-10)
    minus_di_1d = 100 * minus_dm_1d / (atr_1d + 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK + 1, VOL_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]):
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
        
        # Determine regime from 1d ADX
        is_trending = adx_1d_aligned[i] >= ADX_TREND_THRESHOLD
        is_ranging = adx_1d_aligned[i] < ADX_TREND_THRESHOLD
        
        # Camarilla-based signals
        long_signal = False
        short_signal = False
        
        if is_ranging and vol_confirmed:
            # In ranging markets: fade at R3/S3 (mean reversion)
            if close[i] <= camarilla_s3_aligned[i]:
                long_signal = True
            elif close[i] >= camarilla_r3_aligned[i]:
                short_signal = True
        elif is_trending and vol_confirmed:
            # In trending markets: breakout continuation at R4/S4
            if close[i] > camarilla_r4_aligned[i]:
                long_signal = True
            elif close[i] < camarilla_s4_aligned[i]:
                short_signal = True
        
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
exp_6971_6h_camarilla1d_pivot_vol_v1
Hypothesis: 6h Camarilla pivot breakout/fade with 1d volume confirmation.
In ranging markets (ADX < 25): fade at R3/S3 levels. In trending markets (ADX >= 25): breakout continuation at R4/S4.
Uses 1d Camarilla pivots calculated from prior 1d OHLC. ADX filter prevents whipsaw in low volatility.
Designed for 6h timeframe to capture ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to regime via ADX.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6971_6h_camarilla1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use prior 1d for Camarilla calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
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
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for prior 1d
    # Camarilla: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, etc.
    rng = high_1d - low_1d
    camarilla_r4 = close_1d + rng * 1.1 / 2
    camarilla_r3 = close_1d + rng * 1.1 / 4
    camarilla_s3 = close_1d - rng * 1.1 / 4
    camarilla_s4 = close_1d - rng * 1.1 / 2
    
    # Align Camarilla levels to 6h (shifted by 1 for completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d ADX for regime filter
    # TR calculation
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM and -DM
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1d = tr_1d.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    plus_dm_1d = pd.Series(up_move).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    minus_dm_1d = pd.Series(down_move).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # DI and ADX
    plus_di_1d = 100 * plus_dm_1d / (atr_1d + 1e-10)
    minus_di_1d = 100 * minus_dm_1d / (atr_1d + 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK + 1, VOL_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]):
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
        
        # Determine regime from 1d ADX
        is_trending = adx_1d_aligned[i] >= ADX_TREND_THRESHOLD
        is_ranging = adx_1d_aligned[i] < ADX_TREND_THRESHOLD
        
        # Camarilla-based signals
        long_signal = False
        short_signal = False
        
        if is_ranging and vol_confirmed:
            # In ranging markets: fade at R3/S3 (mean reversion)
            if close[i] <= camarilla_s3_aligned[i]:
                long_signal = True
            elif close[i] >= camarilla_r3_aligned[i]:
                short_signal = True
        elif is_trending and vol_confirmed:
            # In trending markets: breakout continuation at R4/S4
            if close[i] > camarilla_r4_aligned[i]:
                long_signal = True
            elif close[i] < camarilla_s4_aligned[i]:
                short_signal = True
        
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