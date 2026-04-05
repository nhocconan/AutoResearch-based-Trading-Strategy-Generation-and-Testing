#!/usr/bin/env python3
"""
exp_7034_1h_donchian20_4h_ema1d_vol_v1
Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and 1d EMA200 regime filter, volume confirmation.
In bull markets (price > 1d EMA200): long breakouts only. In bear markets (price < 1d EMA200): short breakouts only.
4h EMA50 provides intermediate-term trend alignment. Volume confirms breakout legitimacy.
Designed for 1h timeframe with tight entries (target: 60-150 trades over 4 years) using multi-timeframe confluence.
Uses session filter (08-20 UTC) to reduce noise. Fixed size 0.20.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7034_1h_donchian20_4h_ema1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
EMA_4H_PERIOD = 50
EMA_1D_PERIOD = 200
MAX_HOLD_BARS = 24  # ~1 day

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_4H_PERIOD, adjust=False, min_periods=EMA_4H_PERIOD).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_1D_PERIOD, adjust=False, min_periods=EMA_1D_PERIOD).mean().values
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_4H_PERIOD, EMA_1D_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
            
        # Only trade during session
        if not in_session[i]:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend direction from 4h EMA50 and 1d EMA200
        # Bull regime: price > 1d EMA200
        bull_regime = close[i] > ema_1d_aligned[i]
        # Bear regime: price < 1d EMA200
        bear_regime = close[i] < ema_1d_aligned[i]
        
        # 4h EMA50 trend alignment
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        # Breakout signals aligned with 4h trend and 1d regime
        long_breakout = bull_regime and uptrend_4h and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = bear_regime and downtrend_4h and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
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
exp_7034_1h_donchian20_4h_ema1d_vol_v1
Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and 1d EMA200 regime filter, volume confirmation.
In bull markets (price > 1d EMA200): long breakouts only. In bear markets (price < 1d EMA200): short breakouts only.
4h EMA50 provides intermediate-term trend alignment. Volume confirms breakout legitimacy.
Designed for 1h timeframe with tight entries (target: 60-150 trades over 4 years) using multi-timeframe confluence.
Uses session filter (08-20 UTC) to reduce noise. Fixed size 0.20.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7034_1h_donchian20_4h_ema1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
EMA_4H_PERIOD = 50
EMA_1D_PERIOD = 200
MAX_HOLD_BARS = 24  # ~1 day

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_4H_PERIOD, adjust=False, min_periods=EMA_4H_PERIOD).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_1D_PERIOD, adjust=False, min_periods=EMA_1D_PERIOD).mean().values
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_4H_PERIOD, EMA_1D_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
            
        # Only trade during session
        if not in_session[i]:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend direction from 4h EMA50 and 1d EMA200
        # Bull regime: price > 1d EMA200
        bull_regime = close[i] > ema_1d_aligned[i]
        # Bear regime: price < 1d EMA200
        bear_regime = close[i] < ema_1d_aligned[i]
        
        # 4h EMA50 trend alignment
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        # Breakout signals aligned with 4h trend and 1d regime
        long_breakout = bull_regime and uptrend_4h and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = bear_regime and downtrend_4h and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
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