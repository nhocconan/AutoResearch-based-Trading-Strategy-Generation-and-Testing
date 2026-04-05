#!/usr/bin/env python3
"""
exp_7042_12h_donchian20_1d_ema_vol_v2
Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
Uses 1d EMA50 as trend filter: only long when price > EMA50, short when price < EMA50.
Volume threshold reduced to 1.5x to increase trade frequency while maintaining quality.
Position size 0.25 to balance risk and return. Designed for 12h TF to achieve 50-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7042_12h_donchian20_1d_ema_vol_v2"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5  # Reduced from 2.0 to increase signals
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 30  # ~30*12h = 15 days
EMA_PERIOD = 50

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align to LTF (12h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
    highest_since_entry = 0.0  # For trailing stop
    lowest_since_entry = 0.0   # For trailing stop
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Update trailing stop levels
        if position == 1:  # long position
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if close[i] <= highest_since_entry - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                highest_since_entry = 0.0
                continue
        elif position == -1:  # short position
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if close[i] >= lowest_since_entry + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                lowest_since_entry = 0.0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend direction from 1d EMA50
        daily_uptrend = close[i] > ema_1d_aligned[i]
        daily_downtrend = close[i] < ema_1d_aligned[i]
        
        # Breakout signals aligned with 1d trend
        long_breakout = daily_uptrend and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = daily_downtrend and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7042_12h_donchian20_1d_ema_vol_v2
Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
Uses 1d EMA50 as trend filter: only long when price > EMA50, short when price < EMA50.
Volume threshold reduced to 1.5x to increase trade frequency while maintaining quality.
Position size 0.25 to balance risk and return. Designed for 12h TF to achieve 50-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7042_12h_donchian20_1d_ema_vol_v2"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5  # Reduced from 2.0 to increase signals
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 30  # ~30*12h = 15 days
EMA_PERIOD = 50

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align to LTF (12h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
    highest_since_entry = 0.0  # For trailing stop
    lowest_since_entry = 0.0   # For trailing stop
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Update trailing stop levels
        if position == 1:  # long position
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if close[i] <= highest_since_entry - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                highest_since_entry = 0.0
                continue
        elif position == -1:  # short position
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if close[i] >= lowest_since_entry + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                lowest_since_entry = 0.0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend direction from 1d EMA50
        daily_uptrend = close[i] > ema_1d_aligned[i]
        daily_downtrend = close[i] < ema_1d_aligned[i]
        
        # Breakout signals aligned with 1d trend
        long_breakout = daily_uptrend and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = daily_downtrend and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>