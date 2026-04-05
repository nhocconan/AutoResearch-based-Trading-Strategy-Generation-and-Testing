#!/usr/bin/env python3
"""
exp_7119_6h_donchian20_1d_pivot_v2
Hypothesis: Refine 6h Donchian(20) breakout with 1d Camarilla pivot logic by tightening entry conditions.
Add volume confirmation AND require price to close beyond pivot levels (not just touch) to reduce false breakouts.
In ranging markets (between R3/S3): fade extremes only on strong volume and reversal candle.
In trending markets (breaks R4/S4): continuation only on strong volume and close beyond level.
Uses discrete position sizing (0.25) and ATR-based stops to manage risk.
Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on all symbols.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7119_6h_donchian20_1d_pivot_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Increased from 1.8 to reduce false signals
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 24  # ~6 days
PIVOT_CLOSE_THRESHOLD = 0.001  # 0.1% close beyond level required

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R3 = pivot + (range_1d * 1.1 / 2)
    S3 = pivot - (range_1d * 1.1 / 2)
    R4 = pivot + (range_1d * 1.1)
    S4 = pivot - (range_1d * 1.1)
    
    # Align to LTF (6h)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]):
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
        
        # Determine market regime based on Camarilla levels
        in_range = (close[i] > S3_aligned[i]) and (close[i] < R3_aligned[i])
        bull_breakout = close[i] > R4_aligned[i]
        bear_breakout = close[i] < S4_aligned[i]
        
        # Fade at extremes in range (R3/S3) - require close beyond level AND reversal candle
        fade_long = False
        fade_short = False
        if in_range and vol_confirmed:
            # Long fade: price at or below S3 with bullish reversal (close > open)
            if close[i] <= S3_aligned[i] * (1 + PIVOT_CLOSE_THRESHOLD) and close[i] > prices['open'].iloc[i]:
                fade_long = True
            # Short fade: price at or above R3 with bearish reversal (close < open)
            if close[i] >= R3_aligned[i] * (1 - PIVOT_CLOSE_THRESHOLD) and close[i] < prices['open'].iloc[i]:
                fade_short = True
        
        # Continuation breakouts - require close beyond level AND volume
        continuation_long = False
        continuation_short = False
        if vol_confirmed:
            # Long breakout: close beyond R4
            if close[i] > R4_aligned[i] * (1 + PIVOT_CLOSE_THRESHOLD):
                continuation_long = True
            # Short breakout: close beyond S4
            if close[i] < S4_aligned[i] * (1 - PIVOT_CLOSE_THRESHOLD):
                continuation_short = True
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long or continuation_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or continuation_short:
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
exp_7119_6h_donchian20_1d_pivot_v2
Hypothesis: Refine 6h Donchian(20) breakout with 1d Camarilla pivot logic by tightening entry conditions.
Add volume confirmation AND require price to close beyond pivot levels (not just touch) to reduce false breakouts.
In ranging markets (between R3/S3): fade extremes only on strong volume and reversal candle.
In trending markets (breaks R4/S4): continuation only on strong volume and close beyond level.
Uses discrete position sizing (0.25) and ATR-based stops to manage risk.
Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on all symbols.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7119_6h_donchian20_1d_pivot_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Increased from 1.8 to reduce false signals
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 24  # ~6 days
PIVOT_CLOSE_THRESHOLD = 0.001  # 0.1% close beyond level required

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R3 = pivot + (range_1d * 1.1 / 2)
    S3 = pivot - (range_1d * 1.1 / 2)
    R4 = pivot + (range_1d * 1.1)
    S4 = pivot - (range_1d * 1.1)
    
    # Align to LTF (6h)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]):
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
        
        # Determine market regime based on Camarilla levels
        in_range = (close[i] > S3_aligned[i]) and (close[i] < R3_aligned[i])
        bull_breakout = close[i] > R4_aligned[i]
        bear_breakout = close[i] < S4_aligned[i]
        
        # Fade at extremes in range (R3/S3) - require close beyond level AND reversal candle
        fade_long = False
        fade_short = False
        if in_range and vol_confirmed:
            # Long fade: price at or below S3 with bullish reversal (close > open)
            if close[i] <= S3_aligned[i] * (1 + PIVOT_CLOSE_THRESHOLD) and close[i] > prices['open'].iloc[i]:
                fade_long = True
            # Short fade: price at or above R3 with bearish reversal (close < open)
            if close[i] >= R3_aligned[i] * (1 - PIVOT_CLOSE_THRESHOLD) and close[i] < prices['open'].iloc[i]:
                fade_short = True
        
        # Continuation breakouts - require close beyond level AND volume
        continuation_long = False
        continuation_short = False
        if vol_confirmed:
            # Long breakout: close beyond R4
            if close[i] > R4_aligned[i] * (1 + PIVOT_CLOSE_THRESHOLD):
                continuation_long = True
            # Short breakout: close beyond S4
            if close[i] < S4_aligned[i] * (1 - PIVOT_CLOSE_THRESHOLD):
                continuation_short = True
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long or continuation_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or continuation_short:
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