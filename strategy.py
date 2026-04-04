#!/usr/bin/env python3
"""
exp_6679_6h_ichimoku_1d_cloud_filter_v1
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1-day cloud filter for trend alignment.
Uses daily Ichimoku cloud (Senkou Span A/B) to determine primary trend direction:
- Price above 1d cloud = bullish bias, only take longs on 6h TK crosses above
- Price below 1d cloud = bearish bias, only take shorts on 6h TK crosses below
- Price inside 1d cloud = ranging, fade TK crosses toward cloud center (mean reversion)
Volume confirmation reduces false signals. Designed for 6h timeframe to capture swings
while minimizing fee drag (~20-40 trades/year expected). Works in bull/bear/range via regime adaptation.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6679_6h_ichimoku_1d_cloud_filter_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD_FAST = 9   # Tenkan-sen period
TK_PERIOD_SLOW = 26  # Kijun-sen period
IK_PERIOD = 52       # Senkou Span B period
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 4  # ~1 day (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).mean() +
                  pd.Series(low_1d).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).mean()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).mean() +
                 pd.Series(low_1d).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).mean()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=IK_PERIOD, min_periods=IK_PERIOD).mean() +
                      pd.Series(low_1d).rolling(window=IK_PERIOD, min_periods=IK_PERIOD).mean()) / 2).shift(26)
    
    # Align HTF Ichimoku to LTF (6h) with shift(1) for completed days only
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Tenkan-sen and Kijun-sen for crossover signals
    tenkan_6h = (pd.Series(high).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).mean() +
                 pd.Series(low).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).mean()) / 2
    kijun_6h = (pd.Series(high).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).mean() +
                pd.Series(low).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).mean()) / 2
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(TK_PERIOD_SLOW, VOL_MA_PERIOD, ATR_PERIOD, IK_PERIOD) + 26
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
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
            
        # Determine cloud boundaries and position
        upper_cloud = np.maximum(span_a_aligned[i], span_b_aligned[i])
        lower_cloud = np.minimum(span_a_aligned[i], span_b_aligned[i])
        cloud_middle = (upper_cloud + lower_cloud) / 2
        
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        price_in_cloud = (close[i] >= lower_cloud) and (close[i] <= upper_cloud)
        
        # 6h TK crossover signals
        tk_cross_above = (tenkan_6h[i] > kijun_6h[i]) and (tenkan_6h[i-1] <= kijun_6h[i-1])
        tk_cross_below = (tenkan_6h[i] < kijun_6h[i]) and (tenkan_6h[i-1] >= kijun_6h[i-1])
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trading bias based on 1d cloud
        if price_above_cloud:
            # Bullish bias: only take longs on TK cross above
            long_signal = tk_cross_above and vol_confirmed
            short_signal = False  # No shorts in bullish regime
        elif price_below_cloud:
            # Bearish bias: only take shorts on TK cross below
            long_signal = False  # No longs in bearish regime
            short_signal = tk_cross_below and vol_confirmed
        else:
            # Ranging (in cloud): fade TK crosses toward cloud center (mean reversion)
            long_signal = tk_cross_above and (close[i] < cloud_middle) and vol_confirmed
            short_signal = tk_cross_below and (close[i] > cloud_middle) and vol_confirmed
        
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