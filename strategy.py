#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1w EMA trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) upper band, price > 1w EMA(50), volume > 1.5x 20-period average
# Enter short when: price breaks below Donchian(20) lower band, price < 1w EMA(50), volume > 1.5x 20-period average
# Exit when price returns to Donchian middle band or opposite breakout occurs
# Uses 1w trend to filter false breakouts, volume to confirm strength
# Target: 75-150 trades over 4 years by combining trend filter with breakout logic

name = "12h_donchian_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price below Donchian middle OR opposite breakout
            if close[i] < donchian_middle[i] or close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian middle OR opposite breakout
            if close[i] > donchian_middle[i] or close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + 1w trend + volume
            if close[i] > donchian_upper[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bullish breakout with trend and volume confirmation
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_lower[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bearish breakout with trend and volume confirmation
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1w EMA trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) upper band, price > 1w EMA(50), volume > 1.5x 20-period average
# Enter short when: price breaks below Donchian(20) lower band, price < 1w EMA(50), volume > 1.5x 20-period average
# Exit when price returns to Donchian middle band or opposite breakout occurs
# Uses 1w trend to filter false breakouts, volume to confirm strength
# Target: 75-150 trades over 4 years by combining trend filter with breakout logic

name = "12h_donchian_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price below Donchian middle OR opposite breakout
            if close[i] < donchian_middle[i] or close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian middle OR opposite breakout
            if close[i] > donchian_middle[i] or close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + 1w trend + volume
            if close[i] > donchian_upper[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bullish breakout with trend and volume confirmation
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_lower[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bearish breakout with trend and volume confirmation
                signals[i] = -0.25
                position = -1
    
    return signals