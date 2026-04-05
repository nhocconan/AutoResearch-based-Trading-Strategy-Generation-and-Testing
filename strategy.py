#!/usr/bin/env python3
"""
Experiment #11011: 6h Ichimoku Cloud with 1d Trend Filter
Hypothesis: Ichimoku (Tenkan/Kijun cross + price above/below cloud) captures trend momentum.
Using 1d cloud for higher timeframe bias reduces whipsaws. Works in bull (buy above cloud) and
bear (sell below cloud) by using 1d trend filter. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11011_6h_ichimoku1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(KUMO_SHIFT)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(-KUMO_SHIFT)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku for trend filter
    tenkan_daily, kijun_daily, senkou_a_daily, senkou_b_daily, chikou_daily = calculate_ichimoku(
        df_daily['high'].values, df_daily['low'].values, df_daily['close'].values)
    
    tenkan_daily_aligned = align_htf_to_ltf(prices, df_daily, tenkan_daily)
    kijun_daily_aligned = align_htf_to_ltf(prices, df_daily, kijun_daily)
    senkou_a_daily_aligned = align_htf_to_ltf(prices, df_daily, senkou_a_daily)
    senkou_b_daily_aligned = align_htf_to_ltf(prices, df_daily, senkou_b_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_B_PERIOD + KUMO_SHIFT, KIJUN_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily Ichimoku not available
        if (np.isnan(tenkan_daily_aligned[i]) or np.isnan(kijun_daily_aligned[i]) or 
            np.isnan(senkou_a_daily_aligned[i]) or np.isnan(senkou_b_daily_aligned[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Ichimoku conditions
        # Tenkan/Kijun cross
        tk_cross_up = tenkan[i] > kijun[i] and (i == 0 or tenkan[i-1] <= kijun[i-1])
        tk_cross_down = tenkan[i] < kijun[i] and (i == 0 or tenkan[i-1] >= kijun[i-1])
        
        # Price above/below cloud (using 6h cloud)
        # Note: senkou_a/b are already shifted forward by KUMO_SHIFT periods
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i]) if not (np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])) else False
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i]) if not (np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])) else False
        
        # Trend filter (daily): price vs daily cloud
        price_above_daily_cloud = close[i] > max(tenkan_daily_aligned[i], kijun_daily_aligned[i]) if not (np.isnan(tenkan_daily_aligned[i]) or np.isnan(kijun_daily_aligned[i])) else False
        price_below_daily_cloud = close[i] < min(tenkan_daily_aligned[i], kijun_daily_aligned[i]) if not (np.isnan(tenkan_daily_aligned[i]) or np.isnan(kijun_daily_aligned[i])) else False
        
        # Entry conditions
        long_entry = tk_cross_up and price_above_cloud and price_above_daily_cloud
        short_entry = tk_cross_down and price_below_cloud and price_below_daily_cloud
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
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
Experiment #11011: 6h Ichimoku Cloud with 1d Trend Filter
Hypothesis: Ichimoku (Tenkan/Kijun cross + price above/below cloud) captures trend momentum.
Using 1d cloud for higher timeframe bias reduces whipsaws. Works in bull (buy above cloud) and
bear (sell below cloud) by using 1d trend filter. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11011_6h_ichimoku1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(KUMO_SHIFT)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(-KUMO_SHIFT)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku for trend filter
    tenkan_daily, kijun_daily, senkou_a_daily, senkou_b_daily, chikou_daily = calculate_ichimoku(
        df_daily['high'].values, df_daily['low'].values, df_daily['close'].values)
    
    tenkan_daily_aligned = align_htf_to_ltf(prices, df_daily, tenkan_daily)
    kijun_daily_aligned = align_htf_to_ltf(prices, df_daily, kijun_daily)
    senkou_a_daily_aligned = align_htf_to_ltf(prices, df_daily, senkou_a_daily)
    senkou_b_daily_aligned = align_htf_to_ltf(prices, df_daily, senkou_b_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_B_PERIOD + KUMO_SHIFT, KIJUN_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily Ichimoku not available
        if (np.isnan(tenkan_daily_aligned[i]) or np.isnan(kijun_daily_aligned[i]) or 
            np.isnan(senkou_a_daily_aligned[i]) or np.isnan(senkou_b_daily_aligned[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Ichimoku conditions
        # Tenkan/Kijun cross
        tk_cross_up = tenkan[i] > kijun[i] and (i == 0 or tenkan[i-1] <= kijun[i-1])
        tk_cross_down = tenkan[i] < kijun[i] and (i == 0 or tenkan[i-1] >= kijun[i-1])
        
        # Price above/below cloud (using 6h cloud)
        # Note: senkou_a/b are already shifted forward by KUMO_SHIFT periods
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i]) if not (np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])) else False
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i]) if not (np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])) else False
        
        # Trend filter (daily): price vs daily cloud
        price_above_daily_cloud = close[i] > max(tenkan_daily_aligned[i], kijun_daily_aligned[i]) if not (np.isnan(tenkan_daily_aligned[i]) or np.isnan(kijun_daily_aligned[i])) else False
        price_below_daily_cloud = close[i] < min(tenkan_daily_aligned[i], kijun_daily_aligned[i]) if not (np.isnan(tenkan_daily_aligned[i]) or np.isnan(kijun_daily_aligned[i])) else False
        
        # Entry conditions
        long_entry = tk_cross_up and price_above_cloud and price_above_daily_cloud
        short_entry = tk_cross_down and price_below_cloud and price_below_daily_cloud
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>