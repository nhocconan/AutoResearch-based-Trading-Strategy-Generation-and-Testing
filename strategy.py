#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema_vol_v1"
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
    
    # 1-day EMA(20) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_1d[i] = close_1d[i] * 0.1 + ema_1d[i-1] * 0.9
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-day volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 5:
        vol_ma_1d[4] = np.mean(volume_1d[:5])
        for i in range(5, len(volume_1d)):
            vol_ma_1d[i] = volume_1d[i] * 0.1 + vol_ma_1d[i-1] * 0.9
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4-hour Donchian(20) channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 19)  # Donchian needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.3
        
        # Trend filter: price above EMA for long, below EMA for short
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below Donchian low or stoploss
            if close[i] < donch_low[i] or close[i] < entry_price - 2.5 * np.abs(high[i] - low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian high or stoploss
            if close[i] > donch_high[i] or close[i] > entry_price + 2.5 * np.abs(high[i] - low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: breakout above Donchian high in uptrend
                if close[i] > donch_high[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian low in downtrend
                elif close[i] < donch_low[i] and downtrend:
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

name = "4h_donchian20_1d_ema_vol_v1"
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
    
    # 1-day EMA(20) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_1d[i] = close_1d[i] * 0.1 + ema_1d[i-1] * 0.9
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-day volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 5:
        vol_ma_1d[4] = np.mean(volume_1d[:5])
        for i in range(5, len(volume_1d)):
            vol_ma_1d[i] = volume_1d[i] * 0.1 + vol_ma_1d[i-1] * 0.9
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4-hour Donchian(20) channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 19)  # Donchian needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.3
        
        # Trend filter: price above EMA for long, below EMA for short
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below Donchian low or stoploss
            if close[i] < donch_low[i] or close[i] < entry_price - 2.5 * np.abs(high[i] - low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian high or stoploss
            if close[i] > donch_high[i] or close[i] > entry_price + 2.5 * np.abs(high[i] - low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: breakout above Donchian high in uptrend
                if close[i] > donch_high[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian low in downtrend
                elif close[i] < donch_low[i] and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals