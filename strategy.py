#!/usr/bin/env python3

"""
exp_12467_6h_ichimoku1w_trend_vol_v1
Hypothesis: Ichimoku cloud from weekly timeframe provides strong trend filter for 6s Ichimoku.
- Weekly cloud (Senkou Span A/B) determines major trend direction (bull/bear)
- 6h Tenkan-Kijun cross provides entry timing with momentum
- Volume confirmation ensures institutional participation
- Works in bull via TK crosses above cloud, bear via TK crosses below cloud
- Target: 75-150 total trades over 4 years (19-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12467_6h_ichimoku1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD_FAST = 9   # Tenkan-sen (fast)
TK_PERIOD_SLOW = 26  # Kijun-sen (slow)
TK_PERIOD_SENKOUB = 52  # Senkou Span B period
CHIKOU_SHIFT = 26    # Chikou span lag
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).max()
    period9_low = pd.Series(low).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).max()
    period26_low = pd.Series(low).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=TK_PERIOD_SENKOUB, min_periods=TK_PERIOD_SENKOUB).max()
    period52_low = pd.Series(low).rolling(window=TK_PERIOD_SENKOUB, min_periods=TK_PERIOD_SENKOUB).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou = pd.Series(close)
    
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Ichimoku
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    
    # Align weekly Ichimoku to 6s timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    chikou_1w_aligned = align_htf_to_ltf(prices, df_1w, chikou_1w)
    
    # Calculate 6s indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    tenkan_6s, kijun_6s, senkou_a_6s, senkou_b_6s, chikou_6s = calculate_ichimoku(high, low, close)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (need enough data for Ichimoku calculations)
    start = max(TK_PERIOD_SENKOUB + CHIKOU_SHIFT, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly Ichimoku not available
        if np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Weekly trend filter: price above/below cloud
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # 6s Ichimoku signals
        # Tenkan-Kijun cross
        tk_cross_up = tenkan_6s[i] > kijun_6s[i] and tenkan_6s[i-1] <= kijun_6s[i-1]
        tk_cross_down = tenkan_6s[i] < kijun_6s[i] and tenkan_6s[i-1] >= kijun_6s[i-1]
        
        # Chikou confirmation (price vs 26 periods ago)
        chikou_confirm_long = chikou_6s[i] > close[i - CHIKOU_SHIFT] if i >= CHIKOU_SHIFT else False
        chikou_confirm_short = chikou_6s[i] < close[i - CHIKOU_SHIFT] if i >= CHIKOU_SHIFT else False
        
        # Entry conditions
        long_entry = volume_ok and price_above_cloud and tk_cross_up and chikou_confirm_long
        short_entry = volume_ok and price_below_cloud and tk_cross_down and chikou_confirm_short
        
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