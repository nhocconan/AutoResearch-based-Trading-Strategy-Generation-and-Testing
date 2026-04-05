#!/usr/bin/env python3
"""
exp_7251_6h_donchian20_1d_pivot_v2
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot levels for entry/exit.
- Enter long when price breaks above Donchian high AND closes above R3 pivot
- Enter short when price breaks below Donchian low AND closes below S3 pivot
- Exit when price reaches opposite pivot level (R4/S4) or Donchian midpoint
- Uses 1d Camarilla pivots as dynamic support/resistance for structure
- Volume confirmation filter to avoid false breakouts
- Designed for 6h timeframe to capture medium-term swings with ~12-37 trades/year
- Works in bull markets via continuation breakouts and bear markets via faded extremes
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7251_6h_donchian20_1d_pivot_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous day's pivot
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20  # ~5 days (20 * 6h = 120h = 5d)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day OHLC)
    # Camarilla: R4 = C + ((H-L) * 1.5/2), R3 = C + ((H-L) * 1.25/2), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each day
    rng = high_1d - low_1d
    camarilla_r4 = close_1d + (rng * 1.5 / 2)
    camarilla_r3 = close_1d + (rng * 1.25 / 2)
    camarilla_s3 = close_1d - (rng * 1.25 / 2)
    camarilla_s4 = close_1d - (rng * 1.5 / 2)
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2  # Midpoint between R3/S3
    
    # Align to LTF (6h)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
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
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
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
        
        # Donchian breakout conditions
        donchian_breakout_up = close[i] > highest_high[i]
        donchian_breakout_down = close[i] < lowest_low[i]
        
        # Pivot-based entry conditions
        pivot_long = donchian_breakout_up and (close[i] > r3_1d_aligned[i]) and vol_confirmed
        pivot_short = donchian_breakout_down and (close[i] < s3_1d_aligned[i]) and vol_confirmed
        
        # Pivot-based exit conditions
        exit_long = position == 1 and (close[i] >= r4_1d_aligned[i] or close[i] <= mid_1d_aligned[i])
        exit_short = position == -1 and (close[i] <= s4_1d_aligned[i] or close[i] >= mid_1d_aligned[i])
        
        # Enter new positions only if flat
        if position == 0:
            if pivot_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif pivot_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Check for exit
            if exit_long or exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                # Hold current position
                signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7251_6h_donchian20_1d_pivot_v2
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot levels for entry/exit.
- Enter long when price breaks above Donchian high AND closes above R3 pivot
- Enter short when price breaks below Donchian low AND closes below S3 pivot
- Exit when price reaches opposite pivot level (R4/S4) or Donchian midpoint
- Uses 1d Camarilla pivots as dynamic support/resistance for structure
- Volume confirmation filter to avoid false breakouts
- Designed for 6h timeframe to capture medium-term swings with ~12-37 trades/year
- Works in bull markets via continuation breakouts and bear markets via faded extremes
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7251_6h_donchian20_1d_pivot_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous day's pivot
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20  # ~5 days (20 * 6h = 120h = 5d)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day OHLC)
    # Camarilla: R4 = C + ((H-L) * 1.5/2), R3 = C + ((H-L) * 1.25/2), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each day
    rng = high_1d - low_1d
    camarilla_r4 = close_1d + (rng * 1.5 / 2)
    camarilla_r3 = close_1d + (rng * 1.25 / 2)
    camarilla_s3 = close_1d - (rng * 1.25 / 2)
    camarilla_s4 = close_1d - (rng * 1.5 / 2)
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2  # Midpoint between R3/S3
    
    # Align to LTF (6h)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
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
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
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
        
        # Donchian breakout conditions
        donchian_breakout_up = close[i] > highest_high[i]
        donchian_breakout_down = close[i] < lowest_low[i]
        
        # Pivot-based entry conditions
        pivot_long = donchian_breakout_up and (close[i] > r3_1d_aligned[i]) and vol_confirmed
        pivot_short = donchian_breakout_down and (close[i] < s3_1d_aligned[i]) and vol_confirmed
        
        # Pivot-based exit conditions
        exit_long = position == 1 and (close[i] >= r4_1d_aligned[i] or close[i] <= mid_1d_aligned[i])
        exit_short = position == -1 and (close[i] <= s4_1d_aligned[i] or close[i] >= mid_1d_aligned[i])
        
        # Enter new positions only if flat
        if position == 0:
            if pivot_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif pivot_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Check for exit
            if exit_long or exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                # Hold current position
                signals[i] = position * SIGNAL_SIZE
    
    return signals