#!/usr/bin/env python3
"""
exp_7137_4h_donchian20_1d_ema_vol_v2
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
In trending markets (price > 1d EMA50): only take long breakouts above Donchian upper band.
In ranging markets (price near 1d EMA50): take both long and short breakouts with volume confirmation.
Uses 1d EMA50 for regime filter and 4h volume spike for confirmation.
Designed for 4h timeframe to capture swings with ~19-50 trades/year (75-200 total over 4 years).
Improved from exp_7126 by tightening entry conditions and adding EMA trend filter to reduce false breakouts.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7137_4h_donchian20_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

# Parameters - tightened to reduce trade frequency
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_SPIKE_THRESHOLD = 2.0  # Increased from 1.8 to reduce false signals
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # Increased from 6 to allow trades to develop
EMA_BUFFER = 0.005  # 0.5% buffer around EMA for ranging market detection

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align to LTF (4h)
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
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
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
            
        # Volume confirmation - spike above average
        vol_confirmed = volume[i] > vol_ma[i] * VOL_SPIKE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on price vs 1d EMA50
        ema_price_ratio = close[i] / ema_1d_aligned[i] if ema_1d_aligned[i] > 0 else 1.0
        is_uptrend = ema_price_ratio > (1.0 + EMA_BUFFER)
        is_downtrend = ema_price_ratio < (1.0 - EMA_BUFFER)
        is_ranging = not (is_uptrend or is_downtrend)
        
        # Donchian breakout conditions
        breakout_long = close[i] > highest_high[i]
        breakout_short = close[i] < lowest_low[i]
        
        # Entry logic based on regime
        if position == 0:
            if is_uptrend and breakout_long and vol_confirmed:
                # Only take longs in uptrend
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif is_downtrend and breakout_short and vol_confirmed:
                # Only take shorts in downtrend
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif is_ranging and vol_confirmed:
                # In ranging market, take breakouts in both directions
                if breakout_long:
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
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7137_4h_donchian20_1d_ema_vol_v2
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
In trending markets (price > 1d EMA50): only take long breakouts above Donchian upper band.
In ranging markets (price near 1d EMA50): take both long and short breakouts with volume confirmation.
Uses 1d EMA50 for regime filter and 4h volume spike for confirmation.
Designed for 4h timeframe to capture swings with ~19-50 trades/year (75-200 total over 4 years).
Improved from exp_7126 by tightening entry conditions and adding EMA trend filter to reduce false breakouts.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7137_4h_donchian20_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

# Parameters - tightened to reduce trade frequency
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_SPIKE_THRESHOLD = 2.0  # Increased from 1.8 to reduce false signals
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # Increased from 6 to allow trades to develop
EMA_BUFFER = 0.005  # 0.5% buffer around EMA for ranging market detection

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align to LTF (4h)
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
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
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
            
        # Volume confirmation - spike above average
        vol_confirmed = volume[i] > vol_ma[i] * VOL_SPIKE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on price vs 1d EMA50
        ema_price_ratio = close[i] / ema_1d_aligned[i] if ema_1d_aligned[i] > 0 else 1.0
        is_uptrend = ema_price_ratio > (1.0 + EMA_BUFFER)
        is_downtrend = ema_price_ratio < (1.0 - EMA_BUFFER)
        is_ranging = not (is_uptrend or is_downtrend)
        
        # Donchian breakout conditions
        breakout_long = close[i] > highest_high[i]
        breakout_short = close[i] < lowest_low[i]
        
        # Entry logic based on regime
        if position == 0:
            if is_uptrend and breakout_long and vol_confirmed:
                # Only take longs in uptrend
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif is_downtrend and breakout_short and vol_confirmed:
                # Only take shorts in downtrend
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif is_ranging and vol_confirmed:
                # In ranging market, take breakouts in both directions
                if breakout_long:
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
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals