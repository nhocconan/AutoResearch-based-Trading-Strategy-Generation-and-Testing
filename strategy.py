#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_adx_vol_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Hourly EMA(20) for trend
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h EMA(20) for trend direction (HTF)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1d EMA(50) for long-term trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ADX(14) on 4h for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr = np.zeros(len(close_4h))
    tr[0] = high_4h[0] - low_4h[0]
    for i in range(1, len(close_4h)):
        tr[i] = max(high_4h[i] - low_4h[i], 
                   abs(high_4h[i] - close_4h[i-1]), 
                   abs(low_4h[i] - close_4h[i-1]))
    
    # Directional Movement
    dm_plus = np.zeros(len(close_4h))
    dm_minus = np.zeros(len(close_4h))
    for i in range(1, len(close_4h)):
        dm_plus[i] = max(high_4h[i] - high_4h[i-1], 0)
        dm_minus[i] = max(low_4h[i-1] - low_4h[i], 0)
    
    # Smoothing (14-period)
    atr_14 = np.zeros(len(close_4h))
    dm_plus_s = np.zeros(len(close_4h))
    dm_minus_s = np.zeros(len(close_4h))
    
    if len(close_4h) >= 14:
        atr_14[13] = np.sum(tr[0:14])
        dm_plus_s[13] = np.sum(dm_plus[0:14])
        dm_minus_s[13] = np.sum(dm_minus[0:14])
        for i in range(14, len(close_4h)):
            atr_14[i] = atr_14[i-1] - (atr_14[i-1]/14) + tr[i]
            dm_plus_s[i] = dm_plus_s[i-1] - (dm_plus_s[i-1]/14) + dm_plus[i]
            dm_minus_s[i] = dm_minus_s[i-1] - (dm_minus_s[i-1]/14) + dm_minus[i]
    
    # DI and DX
    di_plus = np.zeros(len(close_4h))
    di_minus = np.zeros(len(close_4h))
    dx = np.zeros(len(close_4h))
    
    for i in range(14, len(close_4h)):
        if atr_14[i] != 0:
            di_plus[i] = 100 * dm_plus_s[i] / atr_14[i]
            di_minus[i] = 100 * dm_minus_s[i] / atr_14[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX smoothing
    adx_14 = np.zeros(len(close_4h))
    if len(close_4h) >= 27:
        adx_14[26] = np.mean(dx[13:27]) if np.any(~np.isnan(dx[13:27])) else 0
        for i in range(27, len(close_4h)):
            adx_14[i] = (adx_14[i-1] * 13 + dx[i]) / 14
    
    adx_14_aligned = align_htf_to_ltf(prices, df_4h, adx_14)
    
    # Volume spike detection (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(27, 20, 20, 50)
    
    for i in range(start, n):
        # Skip if session filter fails
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long
            # Exit: trend reversal or stoploss
            if (ema20[i] < ema20_4h_aligned[i] or 
                close[i] < entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short
            # Exit: trend reversal or stoploss
            if (ema20[i] > ema20_4h_aligned[i] or 
                close[i] > entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: EMA20 > EMA20_4h (uptrend) + ADX > 25 + volume spike + above EMA50_1d
            if (ema20[i] > ema20_4h_aligned[i] and 
                adx_14_aligned[i] > 25 and 
                vol_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: EMA20 < EMA20_4h (downtrend) + ADX > 25 + volume spike + below EMA50_1d
            elif (ema20[i] < ema20_4h_aligned[i] and 
                  adx_14_aligned[i] > 25 and 
                  vol_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
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

name = "1h_ema_adx_vol_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Hourly EMA(20) for trend
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h EMA(20) for trend direction (HTF)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1d EMA(50) for long-term trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ADX(14) on 4h for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr = np.zeros(len(close_4h))
    tr[0] = high_4h[0] - low_4h[0]
    for i in range(1, len(close_4h)):
        tr[i] = max(high_4h[i] - low_4h[i], 
                   abs(high_4h[i] - close_4h[i-1]), 
                   abs(low_4h[i] - close_4h[i-1]))
    
    # Directional Movement
    dm_plus = np.zeros(len(close_4h))
    dm_minus = np.zeros(len(close_4h))
    for i in range(1, len(close_4h)):
        dm_plus[i] = max(high_4h[i] - high_4h[i-1], 0)
        dm_minus[i] = max(low_4h[i-1] - low_4h[i], 0)
    
    # Smoothing (14-period)
    atr_14 = np.zeros(len(close_4h))
    dm_plus_s = np.zeros(len(close_4h))
    dm_minus_s = np.zeros(len(close_4h))
    
    if len(close_4h) >= 14:
        atr_14[13] = np.sum(tr[0:14])
        dm_plus_s[13] = np.sum(dm_plus[0:14])
        dm_minus_s[13] = np.sum(dm_minus[0:14])
        for i in range(14, len(close_4h)):
            atr_14[i] = atr_14[i-1] - (atr_14[i-1]/14) + tr[i]
            dm_plus_s[i] = dm_plus_s[i-1] - (dm_plus_s[i-1]/14) + dm_plus[i]
            dm_minus_s[i] = dm_minus_s[i-1] - (dm_minus_s[i-1]/14) + dm_minus[i]
    
    # DI and DX
    di_plus = np.zeros(len(close_4h))
    di_minus = np.zeros(len(close_4h))
    dx = np.zeros(len(close_4h))
    
    for i in range(14, len(close_4h)):
        if atr_14[i] != 0:
            di_plus[i] = 100 * dm_plus_s[i] / atr_14[i]
            di_minus[i] = 100 * dm_minus_s[i] / atr_14[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX smoothing
    adx_14 = np.zeros(len(close_4h))
    if len(close_4h) >= 27:
        adx_14[26] = np.mean(dx[13:27]) if np.any(~np.isnan(dx[13:27])) else 0
        for i in range(27, len(close_4h)):
            adx_14[i] = (adx_14[i-1] * 13 + dx[i]) / 14
    
    adx_14_aligned = align_htf_to_ltf(prices, df_4h, adx_14)
    
    # Volume spike detection (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(27, 20, 20, 50)
    
    for i in range(start, n):
        # Skip if session filter fails
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long
            # Exit: trend reversal or stoploss
            if (ema20[i] < ema20_4h_aligned[i] or 
                close[i] < entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short
            # Exit: trend reversal or stoploss
            if (ema20[i] > ema20_4h_aligned[i] or 
                close[i] > entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: EMA20 > EMA20_4h (uptrend) + ADX > 25 + volume spike + above EMA50_1d
            if (ema20[i] > ema20_4h_aligned[i] and 
                adx_14_aligned[i] > 25 and 
                vol_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: EMA20 < EMA20_4h (downtrend) + ADX > 25 + volume spike + below EMA50_1d
            elif (ema20[i] < ema20_4h_aligned[i] and 
                  adx_14_aligned[i] > 25 and 
                  vol_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals