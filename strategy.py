#!/usr/bin/env python3
"""
6h_12h_1d_price_action_momentum_v1
Strategy: 6h price action momentum with 12h trend filter and 1-day volatility regime
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines 6h price action (close > open for momentum) with 12h EMA trend filter and 1-day low volatility regime (ATR ratio < 0.6) to capture momentum moves in both bull and bear markets. Low volatility regime reduces false breakouts during choppy periods, while EMA filter ensures trades align with higher timeframe trend. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_price_action_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h EMA for momentum (fast)
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    
    # 6h EMA for trend (slow)
    ema_slow = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # 12h EMA trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1-day ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1-day ATR ratio: current ATR / 50-period average ATR
    atr_ma_50_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / atr_ma_50_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Session filter: 00-24 UTC (all hours for 6h timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend condition: 12h EMA slope (using 3-bar change)
        ema_12h_slope = ema_12h_aligned[i] - ema_12h_aligned[i-3]
        uptrend = ema_12h_slope > 0
        downtrend = ema_12h_slope < 0
        
        # Momentum condition: 6h fast EMA > slow EMA
        bullish_momentum = ema_fast[i] > ema_slow[i]
        bearish_momentum = ema_fast[i] < ema_slow[i]
        
        # Price action: strong candle (close > open for bullish, close < open for bearish)
        strong_bullish = close[i] > open_price[i]
        strong_bearish = close[i] < open_price[i]
        
        # Volatility filter: low volatility regime (ATR ratio < 0.6)
        low_volatility = atr_ratio_1d_aligned[i] < 0.6
        
        # Long conditions: uptrend + bullish momentum + strong bullish candle + low volatility
        long_signal = uptrend and bullish_momentum and strong_bullish and low_volatility
        
        # Short conditions: downtrend + bearish momentum + strong bearish candle + low volatility
        short_signal = downtrend and bearish_momentum and strong_bearish and low_volatility
        
        # Exit conditions: momentum reversal or volatility expansion
        exit_long = position == 1 and (not bullish_momentum or not low_volatility)
        exit_short = position == -1 and (not bearish_momentum or not low_volatility)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Combines 6h price action momentum with 12h EMA trend filter and 1-day low volatility regime to capture momentum moves in both bull and bear markets. Low volatility regime reduces false breakouts during choppy periods, while EMA filter ensures trades align with higher timeframe trend. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
EOF
#!/usr/bin/env python3
"""
6h_12h_1d_price_action_momentum_v1
Strategy: 6h price action momentum with 12h trend filter and 1-day volatility regime
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines 6h price action (close > open for momentum) with 12h EMA trend filter and 1-day low volatility regime (ATR ratio < 0.6) to capture momentum moves in both bull and bear markets. Low volatility regime reduces false breakouts during choppy periods, while EMA filter ensures trades align with higher timeframe trend. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_price_action_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h EMA for momentum (fast)
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    
    # 6h EMA for trend (slow)
    ema_slow = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # 12h EMA trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1-day ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1-day ATR ratio: current ATR / 50-period average ATR
    atr_ma_50_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / atr_ma_50_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Session filter: 00-24 UTC (all hours for 6h timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend condition: 12h EMA slope (using 3-bar change)
        ema_12h_slope = ema_12h_aligned[i] - ema_12h_aligned[i-3]
        uptrend = ema_12h_slope > 0
        downtrend = ema_12h_slope < 0
        
        # Momentum condition: 6h fast EMA > slow EMA
        bullish_momentum = ema_fast[i] > ema_slow[i]
        bearish_momentum = ema_fast[i] < ema_slow[i]
        
        # Price action: strong candle (close > open for bullish, close < open for bearish)
        strong_bullish = close[i] > open_price[i]
        strong_bearish = close[i] < open_price[i]
        
        # Volatility filter: low volatility regime (ATR ratio < 0.6)
        low_volatility = atr_ratio_1d_aligned[i] < 0.6
        
        # Long conditions: uptrend + bullish momentum + strong bullish candle + low volatility
        long_signal = uptrend and bullish_momentum and strong_bullish and low_volatility
        
        # Short conditions: downtrend + bearish momentum + strong bearish candle + low volatility
        short_signal = downtrend and bearish_momentum and strong_bearish and low_volatility
        
        # Exit conditions: momentum reversal or volatility expansion
        exit_long = position == 1 and (not bullish_momentum or not low_volatility)
        exit_short = position == -1 and (not bearish_momentum or not low_volatility)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Combines 6h price action momentum with 12h EMA trend filter and 1-day low volatility regime to capture momentum moves in both bull and bear markets. Low volatility regime reduces false breakouts during choppy periods, while EMA filter ensures trades align with higher timeframe trend. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
EOF
#!/usr/bin/env python3
"""
6h_12h_1d_price_action_momentum_v1
Strategy: 6h price action momentum with 12h trend filter and 1-day volatility regime
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines 6h price action (close > open for momentum) with 12h EMA trend filter and 1-day low volatility regime (ATR ratio < 0.6) to capture momentum moves in both bull and bear markets. Low volatility regime reduces false breakouts during choppy periods, while EMA filter ensures trades align with higher timeframe trend. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_price_action_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h EMA for momentum (fast)
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    
    # 6h EMA for trend (slow)
    ema_slow = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # 12h EMA trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1-day ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1-day ATR ratio: current ATR / 50-period average ATR
    atr_ma_50_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / atr_ma_50_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Session filter: 00-24 UTC (all hours for 6h timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend condition: 12h EMA slope (using 3-bar change)
        ema_12h_slope = ema_12h_aligned[i] - ema_12h_aligned[i-3]
        uptrend = ema_12h_slope > 0
        downtrend = ema_12h_slope < 0
        
        # Momentum condition: 6h fast EMA > slow EMA
        bullish_momentum = ema_fast[i] > ema_slow[i]
        bearish_momentum = ema_fast[i] < ema_slow[i]
        
        # Price action: strong candle (close > open for bullish, close < open for bearish)
        strong_bullish = close[i] > open_price[i]
        strong_bearish = close[i] < open_price[i]
        
        # Volatility filter: low volatility regime (ATR ratio < 0.6)
        low_volatility = atr_ratio_1d_aligned[i] < 0.6
        
        # Long conditions: uptrend + bullish momentum + strong bullish candle + low volatility
        long_signal = uptrend and bullish_momentum and strong_bullish and low_volatility
        
        # Short conditions: downtrend + bearish momentum + strong bearish candle + low volatility
        short_signal = downtrend and bearish_momentum and strong_bearish and low_volatility
        
        # Exit conditions: momentum reversal or volatility expansion
        exit_long = position == 1 and (not bullish_momentum or not low_volatility)
        exit_short = position == -1 and (not bearish_momentum or not low_volatility)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Combines 6h price action momentum with 12h EMA trend filter and 1-day low volatility regime to capture momentum moves in both bull and bear markets. Low volatility regime reduces false breakouts during choppy periods, while EMA filter ensures trades align with higher timeframe trend. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
EOF
#!/usr/bin/env python3
"""
6h_12h_1d_price_action_momentum_v1
Strategy: 6h price action momentum with 12h trend filter and 1-day volatility regime
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines 6h price action (close > open for momentum) with 12h EMA trend filter and 1-day low volatility regime (ATR ratio < 0.6) to capture momentum moves in both bull and bear markets. Low volatility regime reduces false breakouts during choppy periods, while EMA filter ensures trades align with higher timeframe trend. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_price_action_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h EMA for momentum (fast)
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    
    # 6h EMA for trend (slow)
    ema_slow = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # 12h EMA trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1-day ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1-day ATR ratio: current ATR / 50-period average ATR
    atr_ma_50_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / atr_ma_50_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Session filter: 00-24 UTC (all hours for 6h timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend condition: 12h EMA slope (using 3-bar change)
        ema_12h_slope = ema_12h_aligned[i] - ema_12h_aligned[i-3]
        uptrend = ema_12h_slope > 0
        downtrend = ema_12h_slope < 0
        
        # Momentum condition: 6h fast EMA > slow EMA
        bullish_momentum = ema_fast[i] > ema_slow[i]
        bearish_momentum = ema_fast[i] < ema_slow[i]
        
        # Price action: strong candle (close > open for bullish, close < open for bearish)
        strong_bullish = close[i] > open_price[i]
        strong_bearish = close[i] < open_price[i]
        
        # Volatility filter: low volatility regime (ATR ratio < 0.6)
        low_volatility = atr_ratio_1d_aligned[i] < 0.6
        
        # Long conditions: uptrend + bullish momentum + strong bullish candle + low volatility
        long_signal = uptrend and bullish_momentum and strong_bullish and low_volatility
        
        # Short conditions: downtrend + bearish momentum + strong bearish candle + low volatility
        short_signal = downtrend and bearish_momentum and strong_bearish and low_volatility
        
        # Exit conditions: momentum reversal or volatility expansion
        exit_long = position == 1 and (not bullish_momentum or not low_volatility)
        exit_short = position == -1 and (not bearish_momentum or not low_volatility)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Combines 6h price action momentum with 12h EMA trend filter and 1-day low volatility regime to capture momentum moves in both bull and bear markets. Low volatility regime reduces false breakouts during choppy periods, while EMA filter ensures trades align with higher timeframe trend. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
EOF
#!/usr/bin/env python3
"""
6h_12h_1d_price_action_momentum_v1
Strategy: 6h price action momentum with 12h trend filter and 1-day volatility regime
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines 6h price action (close > open for momentum) with 12h EMA trend filter and 1-day low volatility regime (ATR ratio < 0.6) to capture momentum moves in both bull and bear markets. Low volatility regime reduces false breakouts during choppy periods, while EMA filter ensures trades align with higher timeframe trend. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_price_action_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h EMA for momentum (fast)
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    
    # 6h EMA for trend (slow)
    ema_slow = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # 12h EMA trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1-day ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1-day ATR ratio: current ATR / 50-period average ATR
    atr_ma_50_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / atr_ma_50_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Session filter: 00-24 UTC (all hours for 6h timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend condition: 12h EMA slope (using 3-bar