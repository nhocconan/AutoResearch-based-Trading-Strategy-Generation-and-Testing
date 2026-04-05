# 1. Hypothesis
# 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation.
# In bull markets (price > 1d VWAP): buy Donchian breakouts above upper band.
# In bear markets (price < 1d VWAP): sell Donchian breakdowns below lower band.
# Uses volume spike (volume > 1.5x 20-period average) to confirm breakout strength.
# Targets 100-200 total trades over 4 years (25-50/year) with strict breakout conditions.
# Works in both bull and bear by aligning with higher-timeframe trend via VWAP.

#!/usr/bin/env python3
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7511_6h_donchian20_1d_vwap_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VWAP_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d VWAP for regime filter
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_array = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_array)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    volume_ma = pd.Series(volume).rolling(window=VWAP_PERIOD, min_periods=VWAP_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VWAP_PERIOD, ATR_PERIOD)
    
    for i in range(start, n):
        # Skip if VWAP not available
        if np.isnan(vwap_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime using VWAP
        above_vwap = close[i] > vwap_1d_aligned[i]  # bull regime
        below_vwap = close[i] < vwap_1d_aligned[i]  # bear regime
        
        # Breakout conditions
        upper_breakout = high[i] > highest_high[i-1]  # using previous bar's channel
        lower_breakout = low[i] < lowest_low[i-1]
        volume_spike = volume[i] > VOLUME_MULTIPLIER * volume_ma[i]
        
        # Entry conditions
        long_entry = above_vwap and upper_breakout and volume_spike
        short_entry = below_vwap and lower_breakout and volume_spike
        
        # Exit conditions (mean reversion to VWAP)
        long_exit = close[i] < vwap_1d_aligned[i]
        short_exit = close[i] > vwap_1d_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
# 1. Hypothesis
# 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation.
# In bull markets (price > 1d VWAP): buy Donchian breakouts above upper band.
# In bear markets (price < 1d VWAP): sell Donchian breakdowns below lower band.
# Uses volume spike (volume > 1.5x 20-period average) to confirm breakout strength.
# Targets 100-200 total trades over 4 years (25-50/year) with strict breakout conditions.
# Works in both bull and bear by aligning with higher-timeframe trend via VWAP.

#!/usr/bin/env python3
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7511_6h_donchian20_1d_vwap_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VWAP_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d VWAP for regime filter
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_array = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_array)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    volume_ma = pd.Series(volume).rolling(window=VWAP_PERIOD, min_periods=VWAP_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VWAP_PERIOD, ATR_PERIOD)
    
    for i in range(start, n):
        # Skip if VWAP not available
        if np.isnan(vwap_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime using VWAP
        above_vwap = close[i] > vwap_1d_aligned[i]  # bull regime
        below_vwap = close[i] < vwap_1d_aligned[i]  # bear regime
        
        # Breakout conditions
        upper_breakout = high[i] > highest_high[i-1]  # using previous bar's channel
        lower_breakout = low[i] < lowest_low[i-1]
        volume_spike = volume[i] > VOLUME_MULTIPLIER * volume_ma[i]
        
        # Entry conditions
        long_entry = above_vwap and upper_breakout and volume_spike
        short_entry = below_vwap and lower_breakout and volume_spike
        
        # Exit conditions (mean reversion to VWAP)
        long_exit = close[i] < vwap_1d_aligned[i]
        short_exit = close[i] > vwap_1d_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals