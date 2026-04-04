#!/usr/bin/env python3
"""
exp_6512_12h_donchian20_1d_vol_v1
Hypothesis: 12h Donchian(20) breakout with volume confirmation.
Uses price channel breakouts as primary signal in both bull and bear markets.
Volume confirmation ensures breakouts have conviction, reducing false signals.
Designed for low-frequency, high-conviction trades on 12h timeframe.
Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6512_12h_donchian20_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0  # volume must be 2.0x its 20-period MA for confirmation
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Long conditions: breaks above Donchian HIGH + volume spike
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: breaks below Donchian LOW + volume spike
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: midpoint reversal
        if position == 1:  # long position
            # Exit if price drops below midpoint of channel
            if close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above midpoint of channel
            if close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_6512_12h_donchian20_1d_vol_v1
Hypothesis: 12h Donchian(20) breakout with volume confirmation.
Uses price channel breakouts as primary signal in both bull and bear markets.
Volume confirmation ensures breakouts have conviction, reducing false signals.
Designed for low-frequency, high-conviction trades on 12h timeframe.
Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6512_12h_donchian20_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0  # volume must be 2.0x its 20-period MA for confirmation
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Long conditions: breaks above Donchian HIGH + volume spike
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: breaks below Donchian LOW + volume spike
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: midpoint reversal
        if position == 1:  # long position
            # Exit if price drops below midpoint of channel
            if close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above midpoint of channel
            if close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>