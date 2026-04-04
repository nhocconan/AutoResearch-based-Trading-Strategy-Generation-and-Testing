#!/usr/bin/env python3
"""
exp_6591_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot direction filter and volume confirmation.
Uses 6h primary timeframe (target: 50-150 total trades over 4 years). 1d Camarilla pivots provide
intraday support/resistance levels derived from previous day's range. Fade at R3/S3, breakout continuation
at R4/S4. Volume confirmation ensures breakouts have conviction. Works in both bull and bear markets
by using pivot levels as dynamic structure that adapts to volatility.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6591_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Volume threshold for confirmation
SIGNAL_SIZE = 0.25      # 25% position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5  # Stoploss at 2.5 * ATR
MAX_HOLD_BARS = 20      # Max hold: ~20 * 6h = 5 days

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # R2 = C + (H - L) * 1.1/6
    # R3 = C + (H - L) * 1.1/4
    # R4 = C + (H - L) * 1.1/2
    # S1 = C - (H - L) * 1.1/12
    # S2 = C - (H - L) * 1.1/6
    # S3 = C - (H - L) * 1.1/4
    # S4 = C - (H - L) * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    r2 = close_1d + (high_1d - low_1d) * 1.1 / 6.0
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    r4 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    s2 = close_1d - (high_1d - low_1d) * 1.1 / 6.0
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    s4 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align to LTF (6h) with shift(1) for completed bars only
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
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
            
        # Determine pivot-based bias
        # Price between S3 and R3: neutral/range (favor mean reversion at extremes)
        # Price > R4: strong uptrend (favor breakout continuation)
        # Price < S4: strong downtrend (favor breakdown continuation)
        # Price > R3 and < R4: fading zone (favor short)
        # Price < S3 and > S4: fading zone (favor long)
        
        # Long conditions:
        # 1. Break above Donchian HIGH (breakout)
        # 2. Volume confirmation
        # 3. Either: price > R4 (breakout continuation) OR price < S3 (fade at support)
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        long_continuation = close[i] > r4_aligned[i]
        long_fade = close[i] < s3_aligned[i]
        
        # Short conditions:
        # 1. Break below Donchian LOW (breakdown)
        # 2. Volume confirmation
        # 3. Either: price < S4 (breakdown continuation) OR price > R3 (fade at resistance)
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        short_continuation = close[i] < s4_aligned[i]
        short_fade = close[i] > r3_aligned[i]
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_volume and (long_continuation or long_fade):
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout and short_volume and (short_continuation or short_fade):
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
exp_6591_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot direction filter and volume confirmation.
Uses 6h primary timeframe (target: 50-150 total trades over 4 years). 1d Camarilla pivots provide
intraday support/resistance levels derived from previous day's range. Fade at R3/S3, breakout continuation
at R4/S4. Volume confirmation ensures breakouts have conviction. Works in both bull and bear markets
by using pivot levels as dynamic structure that adapts to volatility.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6591_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Volume threshold for confirmation
SIGNAL_SIZE = 0.25      # 25% position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5  # Stoploss at 2.5 * ATR
MAX_HOLD_BARS = 20      # Max hold: ~20 * 6h = 5 days

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # R2 = C + (H - L) * 1.1/6
    # R3 = C + (H - L) * 1.1/4
    # R4 = C + (H - L) * 1.1/2
    # S1 = C - (H - L) * 1.1/12
    # S2 = C - (H - L) * 1.1/6
    # S3 = C - (H - L) * 1.1/4
    # S4 = C - (H - L) * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    r2 = close_1d + (high_1d - low_1d) * 1.1 / 6.0
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    r4 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    s2 = close_1d - (high_1d - low_1d) * 1.1 / 6.0
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    s4 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align to LTF (6h) with shift(1) for completed bars only
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
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
            
        # Determine pivot-based bias
        # Price between S3 and R3: neutral/range (favor mean reversion at extremes)
        # Price > R4: strong uptrend (favor breakout continuation)
        # Price < S4: strong downtrend (favor breakdown continuation)
        # Price > R3 and < R4: fading zone (favor short)
        # Price < S3 and > S4: fading zone (favor long)
        
        # Long conditions:
        # 1. Break above Donchian HIGH (breakout)
        # 2. Volume confirmation
        # 3. Either: price > R4 (breakout continuation) OR price < S3 (fade at support)
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        long_continuation = close[i] > r4_aligned[i]
        long_fade = close[i] < s3_aligned[i]
        
        # Short conditions:
        # 1. Break below Donchian LOW (breakdown)
        # 2. Volume confirmation
        # 3. Either: price < S4 (breakdown continuation) OR price > R3 (fade at resistance)
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        short_continuation = close[i] < s4_aligned[i]
        short_fade = close[i] > r3_aligned[i]
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_volume and (long_continuation or long_fade):
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout and short_volume and (short_continuation or short_fade):
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