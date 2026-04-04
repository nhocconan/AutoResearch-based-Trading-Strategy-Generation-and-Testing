#!/usr/bin/env python3
"""
exp_6679_6h_donchian20_12h_trend_vol_v1
Hypothesis: 6h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
Only trade breakouts in the direction of the 12h trend (EMA50 > EMA200 = uptrend, EMA50 < EMA200 = downtrend).
Requires volume > 1.5x 20-period MA for confirmation. Uses ATR-based stops and time exits.
Designed for low trade frequency (target: 15-25/year) with high win rate in both bull/bear markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6679_6h_donchian20_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_FAST = 50
EMA_SLOW = 200
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for EMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMAs for trend filter
    close_12h = df_12h['close'].values
    ema_fast = pd.Series(close_12h).ewm(span=EMA_FAST, adjust=False).mean().values
    ema_slow = pd.Series(close_12h).ewm(span=EMA_SLOW, adjust=False).mean().values
    
    # Align HTF EMAs to LTF (6h) with shift(1) for completed bars only
    ema_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_slow)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_SLOW, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(ema_fast_aligned[i]) or np.isnan(ema_slow_aligned[i]) or 
            np.isnan(highest[i]) or np.isnan(lowest[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
            
        # Determine 12h trend
        uptrend_12h = ema_fast_aligned[i] > ema_slow_aligned[i]
        downtrend_12h = ema_fast_aligned[i] < ema_slow_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest[i-1]  # Break above previous period's high
        breakout_down = close[i] < lowest[i-1]  # Break below previous period's low
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Enter new positions only if flat
        if position == 0:
            # Long: uptrend + upward breakout + volume
            if uptrend_12h and breakout_up and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Short: downtrend + downward breakout + volume
            elif downtrend_12h and breakout_down and vol_confirmed:
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
exp_6679_6h_donchian20_12h_trend_vol_v1
Hypothesis: 6h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
Only trade breakouts in the direction of the 12h trend (EMA50 > EMA200 = uptrend, EMA50 < EMA200 = downtrend).
Requires volume > 1.5x 20-period MA for confirmation. Uses ATR-based stops and time exits.
Designed for low trade frequency (target: 15-25/year) with high win rate in both bull/bear markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6679_6h_donchian20_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_FAST = 50
EMA_SLOW = 200
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for EMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMAs for trend filter
    close_12h = df_12h['close'].values
    ema_fast = pd.Series(close_12h).ewm(span=EMA_FAST, adjust=False).mean().values
    ema_slow = pd.Series(close_12h).ewm(span=EMA_SLOW, adjust=False).mean().values
    
    # Align HTF EMAs to LTF (6h) with shift(1) for completed bars only
    ema_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_slow)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_SLOW, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(ema_fast_aligned[i]) or np.isnan(ema_slow_aligned[i]) or 
            np.isnan(highest[i]) or np.isnan(lowest[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
            
        # Determine 12h trend
        uptrend_12h = ema_fast_aligned[i] > ema_slow_aligned[i]
        downtrend_12h = ema_fast_aligned[i] < ema_slow_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest[i-1]  # Break above previous period's high
        breakout_down = close[i] < lowest[i-1]  # Break below previous period's low
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Enter new positions only if flat
        if position == 0:
            # Long: uptrend + upward breakout + volume
            if uptrend_12h and breakout_up and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Short: downtrend + downward breakout + volume
            elif downtrend_12h and breakout_down and vol_confirmed:
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