#!/usr/bin/env python3
"""
exp_7080_6h_camarilla_pivot_1d_vol_v1
Hypothesis: 6h Camarilla pivot breakout/fade strategy with 1d volume confirmation.
- Fade at R3/S3 levels (mean reversion in ranges)
- Breakout continuation at R4/S4 levels (trend following)
- Uses 1d volume filter to avoid low-liquidity false signals
- Works in both bull/bear markets by adapting to price action at key levels
- Target: 12-37 trades/year (50-150 total over 4 years)
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7080_6h_camarilla_pivot_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 10  # use last 10 days to calculate pivots
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 20  # ~5 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivots and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivots (based on previous day's OHLC)
    # Camarilla levels: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point (standard)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_ * 1.1 / 4)
    r4 = pivot + (range_ * 1.1 / 2)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align pivots to LTF (6h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d volume MA for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need enough for 1d pivots + volume)
    start = max(PIVOT_LOOKBACK + 1, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(r3_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
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
            
        # Volume confirmation (using current 6h volume vs 1d average)
        # Scale 1d volume to 6h equivalent (1d = 4x 6h bars)
        vol_6h_equiv = vol_ma_1d_aligned[i] / 4.0
        vol_confirmed = volume[i] > vol_6h_equiv * VOL_BASE_THRESHOLD if not np.isnan(vol_6h_equiv) else False
        
        # Camarilla-based signals
        # Fade at R3/S3 (mean reversion)
        fade_long = (close[i] <= s3_aligned[i]) and vol_confirmed
        fade_short = (close[i] >= r3_aligned[i]) and vol_confirmed
        
        # Breakout at R4/S4 (continuation)
        breakout_long = (close[i] >= r4_aligned[i]) and vol_confirmed
        breakout_short = (close[i] <= s4_aligned[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
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
exp_7080_6h_camarilla_pivot_1d_vol_v1
Hypothesis: 6h Camarilla pivot breakout/fade strategy with 1d volume confirmation.
- Fade at R3/S3 levels (mean reversion in ranges)
- Breakout continuation at R4/S4 levels (trend following)
- Uses 1d volume filter to avoid low-liquidity false signals
- Works in both bull/bear markets by adapting to price action at key levels
- Target: 12-37 trades/year (50-150 total over 4 years)
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7080_6h_camarilla_pivot_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 10  # use last 10 days to calculate pivots
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 20  # ~5 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivots and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivots (based on previous day's OHLC)
    # Camarilla levels: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point (standard)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_ * 1.1 / 4)
    r4 = pivot + (range_ * 1.1 / 2)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align pivots to LTF (6h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d volume MA for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need enough for 1d pivots + volume)
    start = max(PIVOT_LOOKBACK + 1, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(r3_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
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
            
        # Volume confirmation (using current 6h volume vs 1d average)
        # Scale 1d volume to 6h equivalent (1d = 4x 6h bars)
        vol_6h_equiv = vol_ma_1d_aligned[i] / 4.0
        vol_confirmed = volume[i] > vol_6h_equiv * VOL_BASE_THRESHOLD if not np.isnan(vol_6h_equiv) else False
        
        # Camarilla-based signals
        # Fade at R3/S3 (mean reversion)
        fade_long = (close[i] <= s3_aligned[i]) and vol_confirmed
        fade_short = (close[i] >= r3_aligned[i]) and vol_confirmed
        
        # Breakout at R4/S4 (continuation)
        breakout_long = (close[i] >= r4_aligned[i]) and vol_confirmed
        breakout_short = (close[i] <= s4_aligned[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
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

}