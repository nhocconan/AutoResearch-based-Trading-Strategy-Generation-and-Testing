#!/usr/bin/env python3
"""
exp_6755_6h_ichimoku_cloud_1w_trend_v1
Hypothesis: 6h Ichimoku cloud with 1w trend filter (TK cross + price vs cloud).
In 1w uptrend (price > weekly Senkou Span B): long when price breaks above cloud & TK cross bullish.
In 1w downtrend (price < weekly Senkou Span B): short when price breaks below cloud & TK cross bearish.
Uses cloud as dynamic support/resistance and TK cross for momentum confirmation.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by aligning with weekly Ichimoku trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6755_6h_ichimoku_cloud_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9          # Tenkan-sen period
KJ_PERIOD = 26         # Kijun-sen period
SENKOU_PERIOD = 52     # Senkou span period
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20     # ~5 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for weekly Ichimoku
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Ichimoku components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past TK_PERIOD
    tenkan_1w = (pd.Series(high_1w).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
                 pd.Series(low_1w).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past KJ_PERIOD
    kijun_1w = (pd.Series(high_1w).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
                pd.Series(low_1w).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward KJ_PERIOD
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past SENKOU_PERIOD shifted forward KJ_PERIOD
    senkou_b_1w = ((pd.Series(high_1w).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max() + 
                    pd.Series(low_1w).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min()) / 2)
    # Chikou Span (Lagging Span): close shifted back KJ_PERIOD (not used for signals)
    
    # Align to LTF (6h)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w.values)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w.values)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w.values)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w.values)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # LTF Ichimoku for TK cross
    tenkan_6h = (pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
                 pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
                pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(TK_PERIOD, KJ_PERIOD, SENKOU_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + KJ_PERIOD
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(senkou_b_1w_aligned[i]) or np.isnan(tenkan_1w_aligned[i]) or 
            np.isnan(kijun_1w_aligned[i]) or np.isnan(senkou_a_1w_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend direction from weekly Ichimoku
        # 1w uptrend: price > Senkou Span B
        weekly_uptrend = close[i] > senkou_b_1w_aligned[i]
        # 1w downtrend: price < Senkou Span B
        weekly_downtrend = close[i] < senkou_b_1w_aligned[i]
        
        # TK cross signals
        tk_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # Cloud breakout signals aligned with weekly trend
        # Long: price breaks above cloud (Senkou Span A) in 1w uptrend with bullish TK cross
        long_signal = (weekly_uptrend and 
                      close[i] > senkou_a_1w_aligned[i] and 
                      tk_bullish and 
                      vol_confirmed)
        # Short: price breaks below cloud (Senkou Span A) in 1w downtrend with bearish TK cross
        short_signal = (weekly_downtrend and 
                       close[i] < senkou_a_1w_aligned[i] and 
                       tk_bearish and 
                       vol_confirmed)
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_6755_6h_ichimoku_cloud_1w_trend_v1
Hypothesis: 6h Ichimoku cloud with 1w trend filter (TK cross + price vs cloud).
In 1w uptrend (price > weekly Senkou Span B): long when price breaks above cloud & TK cross bullish.
In 1w downtrend (price < weekly Senkou Span B): short when price breaks below cloud & TK cross bearish.
Uses cloud as dynamic support/resistance and TK cross for momentum confirmation.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by aligning with weekly Ichimoku trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6755_6h_ichimoku_cloud_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9          # Tenkan-sen period
KJ_PERIOD = 26         # Kijun-sen period
SENKOU_PERIOD = 52     # Senkou span period
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20     # ~5 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for weekly Ichimoku
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Ichimoku components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past TK_PERIOD
    tenkan_1w = (pd.Series(high_1w).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
                 pd.Series(low_1w).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past KJ_PERIOD
    kijun_1w = (pd.Series(high_1w).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
                pd.Series(low_1w).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward KJ_PERIOD
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past SENKOU_PERIOD shifted forward KJ_PERIOD
    senkou_b_1w = ((pd.Series(high_1w).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max() + 
                    pd.Series(low_1w).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min()) / 2)
    # Chikou Span (Lagging Span): close shifted back KJ_PERIOD (not used for signals)
    
    # Align to LTF (6h)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w.values)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w.values)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w.values)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w.values)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # LTF Ichimoku for TK cross
    tenkan_6h = (pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
                 pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
                pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(TK_PERIOD, KJ_PERIOD, SENKOU_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + KJ_PERIOD
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(senkou_b_1w_aligned[i]) or np.isnan(tenkan_1w_aligned[i]) or 
            np.isnan(kijun_1w_aligned[i]) or np.isnan(senkou_a_1w_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend direction from weekly Ichimoku
        # 1w uptrend: price > Senkou Span B
        weekly_uptrend = close[i] > senkou_b_1w_aligned[i]
        # 1w downtrend: price < Senkou Span B
        weekly_downtrend = close[i] < senkou_b_1w_aligned[i]
        
        # TK cross signals
        tk_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # Cloud breakout signals aligned with weekly trend
        # Long: price breaks above cloud (Senkou Span A) in 1w uptrend with bullish TK cross
        long_signal = (weekly_uptrend and 
                      close[i] > senkou_a_1w_aligned[i] and 
                      tk_bullish and 
                      vol_confirmed)
        # Short: price breaks below cloud (Senkou Span A) in 1w downtrend with bearish TK cross
        short_signal = (weekly_downtrend and 
                       close[i] < senkou_a_1w_aligned[i] and 
                       tk_bearish and 
                       vol_confirmed)
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals