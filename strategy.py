#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1d data for EMA and Donchian channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA(50) on daily closes
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * (2/51) + ema_50_1d[i-1] * (49/51)
    
    # Donchian channel (20) on daily high/low
    donch_high_1d = np.full(len(high_1d), np.nan)
    donch_low_1d = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_high_1d[i] = np.max(high_1d[i-20:i])
        donch_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Align 1d EMA and Donchian to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Volume filter: current volume > 1.5x average over last 24 periods (24*12h = 12 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 24)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian low or stoploss hit
            if (close[i] < donch_low_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high or stoploss hit
            if (close[i] > donch_high_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above Donchian high with EMA(50) above price and volume
            if (close[i] > donch_high_aligned[i] and ema_50_aligned[i] > close[i] and volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low with EMA(50) below price and volume
            elif (close[i] < donch_low_aligned[i] and ema_50_aligned[i] < close[i] and volume_filter):
                signals[i] = -0.25
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

name = "12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1d data for EMA and Donchian channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA(50) on daily closes
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * (2/51) + ema_50_1d[i-1] * (49/51)
    
    # Donchian channel (20) on daily high/low
    donch_high_1d = np.full(len(high_1d), np.nan)
    donch_low_1d = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_high_1d[i] = np.max(high_1d[i-20:i])
        donch_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Align 1d EMA and Donchian to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Volume filter: current volume > 1.5x average over last 24 periods (24*12h = 12 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 24)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian low or stoploss hit
            if (close[i] < donch_low_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high or stoploss hit
            if (close[i] > donch_high_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above Donchian high with EMA(50) above price and volume
            if (close[i] > donch_high_aligned[i] and ema_50_aligned[i] > close[i] and volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low with EMA(50) below price and volume
            elif (close[i] < donch_low_aligned[i] and ema_50_aligned[i] < close[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals