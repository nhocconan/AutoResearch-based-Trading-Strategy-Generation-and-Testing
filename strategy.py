#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Goes long when price breaks above 20-period high with 1d ADX > 25 (trending) and volume > average.
# Goes short when price breaks below 20-period low with 1d ADX > 25 (trending) and volume > average.
# Uses ATR-based stoploss to limit downside. Designed to work in both bull and bear markets
# by only taking trades in trending markets (ADX > 25) while avoiding range-bound conditions.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6h_donchian20_1d_adx_vol_v1"
timeframe = "6h"
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
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.zeros_like(tr)
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14 if not np.isnan(atr_1d[i-1]) else tr[i]
    atr_1d[0] = tr[0]
    
    # Directional Movement
    up_move = np.zeros_like(high_1d)
    down_move = np.zeros_like(low_1d)
    up_move[0] = 0
    down_move[0] = 0
    for i in range(1, len(high_1d)):
        up_move[i] = max(high_1d[i] - high_1d[i-1], 0)
        down_move[i] = max(low_1d[i-1] - low_1d[i], 0)
    
    # Directional Indicators
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_14 = np.zeros_like(tr)
    plus_dm_14 = np.zeros_like(up_move)
    minus_dm_14 = np.zeros_like(down_move)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.mean(tr[1:15])
            plus_dm_14[i] = np.mean(plus_dm[1:15])
            minus_dm_14[i] = np.mean(minus_dm[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
            plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # Avoid division by zero
    plus_di_14 = np.zeros_like(atr_14)
    minus_di_14 = np.zeros_like(atr_14)
    dx = np.zeros_like(atr_14)
    for i in range(14, len(atr_14)):
        if atr_14[i] > 0:
            plus_di_14[i] = (plus_dm_14[i] / atr_14[i]) * 100
            minus_di_14[i] = (minus_dm_14[i] / atr_14[i]) * 100
            dx[i] = (abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])) * 100
    
    # ADX calculation
    adx_1d = np.zeros_like(dx)
    for i in range(28, len(dx)):  # 28 = 14 + 14
        if i == 28:
            adx_1d[i] = np.mean(dx[14:29])
        else:
            adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels (20-period) on 6h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (6h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or ADX < 20 (range)
            elif close[i] < low_min[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or ADX < 20 (range)
            elif close[i] > high_max[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter and ADX > 25 (trending)
            if vol_filter and adx_1d_aligned[i] > 25:
                # Long entry: price breaks above Donchian high
                if close[i] > high_max[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short entry: price breaks below Donchian low
                elif close[i] < low_min[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Goes long when price breaks above 20-period high with 1d ADX > 25 (trending) and volume > average.
# Goes short when price breaks below 20-period low with 1d ADX > 25 (trending) and volume > average.
# Uses ATR-based stoploss to limit downside. Designed to work in both bull and bear markets
# by only taking trades in trending markets (ADX > 25) while avoiding range-bound conditions.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6h_donchian20_1d_adx_vol_v1"
timeframe = "6h"
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
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.zeros_like(tr)
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14 if not np.isnan(atr_1d[i-1]) else tr[i]
    atr_1d[0] = tr[0]
    
    # Directional Movement
    up_move = np.zeros_like(high_1d)
    down_move = np.zeros_like(low_1d)
    up_move[0] = 0
    down_move[0] = 0
    for i in range(1, len(high_1d)):
        up_move[i] = max(high_1d[i] - high_1d[i-1], 0)
        down_move[i] = max(low_1d[i-1] - low_1d[i], 0)
    
    # Directional Indicators
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_14 = np.zeros_like(tr)
    plus_dm_14 = np.zeros_like(up_move)
    minus_dm_14 = np.zeros_like(down_move)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.mean(tr[1:15])
            plus_dm_14[i] = np.mean(plus_dm[1:15])
            minus_dm_14[i] = np.mean(minus_dm[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
            plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # Avoid division by zero
    plus_di_14 = np.zeros_like(atr_14)
    minus_di_14 = np.zeros_like(atr_14)
    dx = np.zeros_like(atr_14)
    for i in range(14, len(atr_14)):
        if atr_14[i] > 0:
            plus_di_14[i] = (plus_dm_14[i] / atr_14[i]) * 100
            minus_di_14[i] = (minus_dm_14[i] / atr_14[i]) * 100
            dx[i] = (abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])) * 100
    
    # ADX calculation
    adx_1d = np.zeros_like(dx)
    for i in range(28, len(dx)):  # 28 = 14 + 14
        if i == 28:
            adx_1d[i] = np.mean(dx[14:29])
        else:
            adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels (20-period) on 6h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (6h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or ADX < 20 (range)
            elif close[i] < low_min[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or ADX < 20 (range)
            elif close[i] > high_max[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter and ADX > 25 (trending)
            if vol_filter and adx_1d_aligned[i] > 25:
                # Long entry: price breaks above Donchian high
                if close[i] > high_max[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short entry: price breaks below Donchian low
                elif close[i] < low_min[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>