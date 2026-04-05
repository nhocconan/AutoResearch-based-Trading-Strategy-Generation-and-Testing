#!/usr/bin/env python3
"""
exp_7267_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot regime filter for BTC/ETH/SOL.
In trending markets (price between R3/S3): breakout continuation in breakout direction.
In ranging markets (price outside R3/S3): mean reversion at Donchian extremes with volume confirmation.
Uses 1d Camarilla pivots for regime classification and 6h volume for confirmation.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to pivot-defined structure: 
- Trending regime (R3-S3): trade breakouts with trend
- Ranging regime (outside R3-S3): fade extremes
Avoids whipsaw in sideways markets and captures momentum in trending periods.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7267_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8  # ~4 days (8 * 6h = 48h)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_h5 = typical_price + 1.1 * range_1d / 2  # R4
    camarilla_h4 = typical_price + 1.1 * range_1d / 4  # R3
    camarilla_h3 = typical_price + 1.1 * range_1d / 6  # R2
    camarilla_l3 = typical_price - 1.1 * range_1d / 6  # S2
    camarilla_l4 = typical_price - 1.1 * range_1d / 4  # S3
    camarilla_l5 = typical_price - 1.1 * range_1d / 2  # S4
    
    # Align to LTF (6h)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)  # R3
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)  # S3
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)  # R4
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)  # S4
    
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
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]):
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
        # Trending regime: price between R3 and S3
        in_trending_regime = (close[i] > camarilla_l4_aligned[i]) and (close[i] < camarilla_h4_aligned[i])
        # Ranging regime: price outside R3/S3 (beyond R3 or below S3)
        in_ranging_regime = (close[i] <= camarilla_l4_aligned[i]) or (close[i] >= camarilla_h4_aligned[i])
        
        # Fade at Donchian extremes in ranging market
        fade_long = in_ranging_regime and (close[i] <= lowest_low[i]) and vol_confirmed
        fade_short = in_ranging_regime and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Continuation breakouts in trending market (trade with structure)
        continuation_long = in_trending_regime and (close[i] > highest_high[i]) and vol_confirmed
        continuation_short = in_trending_regime and (close[i] < lowest_low[i]) and vol_confirmed
        
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
exp_7267_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot regime filter for BTC/ETH/SOL.
In trending markets (price between R3/S3): breakout continuation in breakout direction.
In ranging markets (price outside R3/S3): mean reversion at Donchian extremes with volume confirmation.
Uses 1d Camarilla pivots for regime classification and 6h volume for confirmation.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to pivot-defined structure: 
- Trending regime (R3-S3): trade breakouts with trend
- Ranging regime (outside R3-S3): fade extremes
Avoids whipsaw in sideways markets and captures momentum in trending periods.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7267_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8  # ~4 days (8 * 6h = 48h)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_h5 = typical_price + 1.1 * range_1d / 2  # R4
    camarilla_h4 = typical_price + 1.1 * range_1d / 4  # R3
    camarilla_h3 = typical_price + 1.1 * range_1d / 6  # R2
    camarilla_l3 = typical_price - 1.1 * range_1d / 6  # S2
    camarilla_l4 = typical_price - 1.1 * range_1d / 4  # S3
    camarilla_l5 = typical_price - 1.1 * range_1d / 2  # S4
    
    # Align to LTF (6h)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)  # R3
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)  # S3
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)  # R4
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)  # S4
    
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
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]):
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
        # Trending regime: price between R3 and S3
        in_trending_regime = (close[i] > camarilla_l4_aligned[i]) and (close[i] < camarilla_h4_aligned[i])
        # Ranging regime: price outside R3/S3 (beyond R3 or below S3)
        in_ranging_regime = (close[i] <= camarilla_l4_aligned[i]) or (close[i] >= camarilla_h4_aligned[i])
        
        # Fade at Donchian extremes in ranging market
        fade_long = in_ranging_regime and (close[i] <= lowest_low[i]) and vol_confirmed
        fade_short = in_ranging_regime and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Continuation breakouts in trending market (trade with structure)
        continuation_long = in_trending_regime and (close[i] > highest_high[i]) and vol_confirmed
        continuation_short = in_trending_regime and (close[i] < lowest_low[i]) and vol_confirmed
        
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