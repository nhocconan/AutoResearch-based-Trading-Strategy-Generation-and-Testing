#!/usr/bin/env python3
"""
Experiment #9595: 6h Ichimoku Cloud + Volume Spike + Multi-Timeframe Trend Filter.
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h, filtered by daily cloud color (bull/bear) and 
weekly trend (from weekly EMA200), with volume spike confirmation, provides high-probability 
trend-following entries. Works in bull markets (long when price above cloud, TK cross up) 
and bear markets (short when price below cloud, TK cross down). Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9595_6h_ichimoku_cloud_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD_FAST = 9   # Tenkan-sen period
TK_PERIOD_SLOW = 26  # Kijun-sen period
SENKOU_SPAN_B_PERIOD = 52
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
EMA200_PERIOD = 200
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return tr

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """
    Calculate Ichimoku components
    Tenkan-sen = (HH9 + LL9) / 2
    Kijun-sen = (HH26 + LL26) / 2
    Senkou Span A = (Tenkan-sen + Kijun-sen) / 2
    Senkou Span B = (HH52 + LL52) / 2
    Chikou Span = close shifted back 26 periods
    """
    # Tenkan-sen (Conversion Line): 9-period high-low midpoint
    high9 = pd.Series(high).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).max()
    low9 = pd.Series(low).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).min()
    tenkan = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): 26-period high-low midpoint
    high26 = pd.Series(high).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).max()
    low26 = pd.Series(low).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).min()
    kijun = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): 52-period high-low midpoint
    high52 = pd.Series(high).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max()
    low52 = pd.Series(low).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()
    senkou_b = (high52 + low52) / 2
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def calculate_ema(values, period):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')   # for Ichimoku and cloud
    df_1w = get_htf_data(prices, '1w')   # for weekly trend filter (EMA200)
    
    # Calculate Ichimoku on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = calculate_ema(close_1w, EMA200_PERIOD)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TK_PERIOD_SLOW, SENKOU_SPAN_B_PERIOD, VOLUME_MA_PERIOD, EMA200_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Ichimoku signals on 6h timeframe using daily Ichimoku
        price_above_cloud = close[i] > max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        tk_cross_up = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_down = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        # Entry conditions
        long_entry = volume_spike and price_above_cloud and tk_cross_up and weekly_uptrend
        short_entry = volume_spike and price_below_cloud and tk_cross_down and weekly_downtrend
        
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
Experiment #9595: 6h Ichimoku Cloud + Volume Spike + Multi-Timeframe Trend Filter.
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h, filtered by daily cloud color (bull/bear) and 
weekly trend (from weekly EMA200), with volume spike confirmation, provides high-probability 
trend-following entries. Works in bull markets (long when price above cloud, TK cross up) 
and bear markets (short when price below cloud, TK cross down). Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9595_6h_ichimoku_cloud_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD_FAST = 9   # Tenkan-sen period
TK_PERIOD_SLOW = 26  # Kijun-sen period
SENKOU_SPAN_B_PERIOD = 52
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
EMA200_PERIOD = 200
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return tr

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """
    Calculate Ichimoku components
    Tenkan-sen = (HH9 + LL9) / 2
    Kijun-sen = (HH26 + LL26) / 2
    Senkou Span A = (Tenkan-sen + Kijun-sen) / 2
    Senkou Span B = (HH52 + LL52) / 2
    Chikou Span = close shifted back 26 periods
    """
    # Tenkan-sen (Conversion Line): 9-period high-low midpoint
    high9 = pd.Series(high).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).max()
    low9 = pd.Series(low).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).min()
    tenkan = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): 26-period high-low midpoint
    high26 = pd.Series(high).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).max()
    low26 = pd.Series(low).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).min()
    kijun = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): 52-period high-low midpoint
    high52 = pd.Series(high).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max()
    low52 = pd.Series(low).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()
    senkou_b = (high52 + low52) / 2
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def calculate_ema(values, period):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')   # for Ichimoku and cloud
    df_1w = get_htf_data(prices, '1w')   # for weekly trend filter (EMA200)
    
    # Calculate Ichimoku on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = calculate_ema(close_1w, EMA200_PERIOD)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TK_PERIOD_SLOW, SENKOU_SPAN_B_PERIOD, VOLUME_MA_PERIOD, EMA200_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Ichimoku signals on 6h timeframe using daily Ichimoku
        price_above_cloud = close[i] > max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        tk_cross_up = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_down = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        # Entry conditions
        long_entry = volume_spike and price_above_cloud and tk_cross_up and weekly_uptrend
        short_entry = volume_spike and price_below_cloud and tk_cross_down and weekly_downtrend
        
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