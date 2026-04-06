#!/usr/bin/env python3
"""
1h Momentum with 4h/1d Trend Filter and Volume Confirmation
Hypothesis: In trending markets, momentum on 1h aligned with 4h/1d trend direction captures moves.
Volume confirmation reduces false signals. Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend).
Target: 60-150 total trades over 4 years = 15-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_4h1d_trend_vol_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data for trend filters (once before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA(50) for trend direction
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA(100) for stronger trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 100  # For EMA and RSI
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ema[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        # Check exits
        if position == 1:  # long position
            # Exit: stoploss or trend reversal
            if (close[i] <= entry_price - 2.5 * atr[i] or
                close[i] < ema_4h_aligned[i] or
                close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: stoploss or trend reversal
            if (close[i] >= entry_price + 2.5 * atr[i] or
                close[i] > ema_4h_aligned[i] or
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: momentum with trend and volume confirmation
            if in_session:
                bullish = (rsi[i] > 55 and rsi[i] < 75 and
                           close[i] > ema_4h_aligned[i] and
                           close[i] > ema_1d_aligned[i] and
                           volume[i] > vol_ema[i] * 1.3)
                bearish = (rsi[i] < 45 and rsi[i] > 25 and
                           close[i] < ema_4h_aligned[i] and
                           close[i] < ema_1d_aligned[i] and
                           volume[i] > vol_ema[i] * 1.3)
                
                if bullish:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                elif bearish:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Momentum with 4h/1d Trend Filter and Volume Confirmation
Hypothesis: In trending markets, momentum on 1h aligned with 4h/1d trend direction captures moves.
Volume confirmation reduces false signals. Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend).
Target: 60-150 total trades over 4 years = 15-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_4h1d_trend_vol_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data for trend filters (once before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA(50) for trend direction
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA(100) for stronger trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 100  # For EMA and RSI
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ema[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        # Check exits
        if position == 1:  # long position
            # Exit: stoploss or trend reversal
            if (close[i] <= entry_price - 2.5 * atr[i] or
                close[i] < ema_4h_aligned[i] or
                close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: stoploss or trend reversal
            if (close[i] >= entry_price + 2.5 * atr[i] or
                close[i] > ema_4h_aligned[i] or
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: momentum with trend and volume confirmation
            if in_session:
                bullish = (rsi[i] > 55 and rsi[i] < 75 and
                           close[i] > ema_4h_aligned[i] and
                           close[i] > ema_1d_aligned[i] and
                           volume[i] > vol_ema[i] * 1.3)
                bearish = (rsi[i] < 45 and rsi[i] > 25 and
                           close[i] < ema_4h_aligned[i] and
                           close[i] < ema_1d_aligned[i] and
                           volume[i] > vol_ema[i] * 1.3)
                
                if bullish:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                elif bearish:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals