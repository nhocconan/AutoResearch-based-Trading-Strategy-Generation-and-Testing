#!/usr/bin/env python3
"""
Experiment #8191: 6-hour Ichimoku Cloud breakout with 1-day trend filter.
Hypothesis: In strong trends (bull/bear), price breaks above/below Kumo (cloud) on 6h 
with Tenkan/Kijun cross confirming momentum. The 1-day trend filter ensures we only 
trade in the direction of the higher timeframe trend, reducing false signals during 
range-bound periods. Ichimoku provides dynamic support/resistance that adapts to 
volatility, working in both bull and bear markets. Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8191_6h_ichimoku1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
ATR_TARGET_MULTIPLIER = 3.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(KUMO_SHIFT)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Current Kumo (cloud) boundaries - use unshifted values for current cloud
    senkou_a_current = (tenkan_sen + kijun_sen) / 2
    senkou_b_current = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                        pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # TK Cross
    tk_cross = tenkan_sen - kijun_sen
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # TK Cross signals
        tk_cross_up = (tk_cross.iloc[i] > 0) and (tk_cross.iloc[i-1] <= 0) if i > 0 else False
        tk_cross_down = (tk_cross.iloc[i] < 0) and (tk_cross.iloc[i-1] >= 0) if i > 0 else False
        
        # Entry conditions
        long_entry = bull_bias and price_above_cloud and tk_cross_up
        short_entry = bear_bias and price_below_cloud and tk_cross_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8191: 6-hour Ichimoku Cloud breakout with 1-day trend filter.
Hypothesis: In strong trends (bull/bear), price breaks above/below Kumo (cloud) on 6h 
with Tenkan/Kijun cross confirming momentum. The 1-day trend filter ensures we only 
trade in the direction of the higher timeframe trend, reducing false signals during 
range-bound periods. Ichimoku provides dynamic support/resistance that adapts to 
volatility, working in both bull and bear markets. Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8191_6h_ichimoku1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
ATR_TARGET_MULTIPLIER = 3.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(KUMO_SHIFT)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Current Kumo (cloud) boundaries - use unshifted values for current cloud
    senkou_a_current = (tenkan_sen + kijun_sen) / 2
    senkou_b_current = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                        pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # TK Cross
    tk_cross = tenkan_sen - kijun_sen
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # TK Cross signals
        tk_cross_up = (tk_cross.iloc[i] > 0) and (tk_cross.iloc[i-1] <= 0) if i > 0 else False
        tk_cross_down = (tk_cross.iloc[i] < 0) and (tk_cross.iloc[i-1] >= 0) if i > 0 else False
        
        # Entry conditions
        long_entry = bull_bias and price_above_cloud and tk_cross_up
        short_entry = bear_bias and price_below_cloud and tk_cross_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8191: 6-hour Ichimoku Cloud breakout with 1-day trend filter.
Hypothesis: In strong trends (bull/bear), price breaks above/below Kumo (cloud) on 6h 
with Tenkan/Kijun cross confirming momentum. The 1-day trend filter ensures we only 
trade in the direction of the higher timeframe trend, reducing false signals during 
range-bound periods. Ichimoku provides dynamic support/resistance that adapts to 
volatility, working in both bull and bear markets. Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8191_6h_ichimoku1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
ATR_TARGET_MULTIPLIER = 3.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(KUMO_SHIFT)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Current Kumo (cloud) boundaries - use unshifted values for current cloud
    senkou_a_current = (tenkan_sen + kijun_sen) / 2
    senkou_b_current = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                        pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # TK Cross
    tk_cross = tenkan_sen - kijun_sen
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # TK Cross signals
        tk_cross_up = (tk_cross.iloc[i] > 0) and (tk_cross.iloc[i-1] <= 0) if i > 0 else False
        tk_cross_down = (tk_cross.iloc[i] < 0) and (tk_cross.iloc[i-1] >= 0) if i > 0 else False
        
        # Entry conditions
        long_entry = bull_bias and price_above_cloud and tk_cross_up
        short_entry = bear_bias and price_below_cloud and tk_cross_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8191: 6-hour Ichimoku Cloud breakout with 1-day trend filter.
Hypothesis: In strong trends (bull/bear), price breaks above/below Kumo (cloud) on 6h 
with Tenkan/Kijun cross confirming momentum. The 1-day trend filter ensures we only 
trade in the direction of the higher timeframe trend, reducing false signals during 
range-bound periods. Ichimoku provides dynamic support/resistance that adapts to 
volatility, working in both bull and bear markets. Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8191_6h_ichimoku1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
ATR_TARGET_MULTIPLIER = 3.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(KUMO_SHIFT)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Current Kumo (cloud) boundaries - use unshifted values for current cloud
    senkou_a_current = (tenkan_sen + kijun_sen) / 2
    senkou_b_current = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                        pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # TK Cross
    tk_cross = tenkan_sen - kijun_sen
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # TK Cross signals
        tk_cross_up = (tk_cross.iloc[i] > 0) and (tk_cross.iloc[i-1] <= 0) if i > 0 else False
        tk_cross_down = (tk_cross.iloc[i] < 0) and (tk_cross.iloc[i-1] >= 0) if i > 0 else False
        
        # Entry conditions
        long_entry = bull_bias and price_above_cloud and tk_cross_up
        short_entry = bear_bias and price_below_cloud and tk_cross_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8191: 6-hour Ichimoku Cloud breakout with 1-day trend filter.
Hypothesis: In strong trends (bull/bear), price breaks above/below Kumo (cloud) on 6h 
with Tenkan/Kijun cross confirming momentum. The 1-day trend filter ensures we only 
trade in the direction of the higher timeframe trend, reducing false signals during 
range-bound periods. Ichimoku provides dynamic support/resistance that adapts to 
volatility, working in both bull and bear markets. Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8191_6h_ichimoku1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
ATR_TARGET_MULTIPLIER = 3.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_