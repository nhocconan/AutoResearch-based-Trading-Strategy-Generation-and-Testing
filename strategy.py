#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema50_vol_v13"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 50-period EMA for trend filter
    ema50 = np.full(n, np.nan)
    if n >= 50:
        multiplier = 2 / (50 + 1)
        ema50[49] = np.mean(close[:50])
        for i in range(50, n):
            ema50[i] = close[i] * multiplier + ema50[i-1] * (1 - multiplier)
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period Donchian channel from previous day
    donch_high_1d = np.full(len(high_1d), np.nan)
    donch_low_1d = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donch_high_1d[i] = np.max(high_1d[i-20:i])
        donch_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian low or stoploss hit
            if (close[i] < donch_low_aligned[i] or
                close[i] < entry_price - 2.0 * ema50[i]):  # Using EMA50 as proxy for volatility
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Exit: price closes above Donchian high or stoploss hit
            if (close[i] > donch_high_aligned[i] or
                close[i] > entry_price + 2.0 * ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            # Look for entries
            # Long: price breaks above Donchian high with volume and above EMA50
            if (close[i] > donch_high_aligned[i] and volume_filter and close[i] > ema50[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low with volume and below EMA50
            elif (close[i] < donch_low_aligned[i] and volume_filter and close[i] < ema50[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema50_vol_v13"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 50-period EMA for trend filter
    ema50 = np.full(n, np.nan)
    if n >= 50:
        multiplier = 2 / (50 + 1)
        ema50[49] = np.mean(close[:50])
        for i in range(50, n):
            ema50[i] = close[i] * multiplier + ema50[i-1] * (1 - multiplier)
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period Donchian channel from previous day
    donch_high_1d = np.full(len(high_1d), np.nan)
    donch_low_1d = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donch_high_1d[i] = np.max(high_1d[i-20:i])
        donch_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian low or stoploss hit
            if (close[i] < donch_low_aligned[i] or
                close[i] < entry_price - 2.0 * ema50[i]):  # Using EMA50 as proxy for volatility
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Exit: price closes above Donchian high or stoploss hit
            if (close[i] > donch_high_aligned[i] or
                close[i] > entry_price + 2.0 * ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            # Look for entries
            # Long: price breaks above Donchian high with volume and above EMA50
            if (close[i] > donch_high_aligned[i] and volume_filter and close[i] > ema50[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low with volume and below EMA50
            elif (close[i] < donch_low_aligned[i] and volume_filter and close[i] < ema50[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals