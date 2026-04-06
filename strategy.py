#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Long: Price > Kumo (cloud) + TK Cross bullish + 1d EMA(50) up + volume > 1.5x average
- Short: Price < Kumo (cloud) + TK Cross bearish + 1d EMA(50) down + volume > 1.5x average
- Exit: Price crosses back into Kumo or reversal signal
- Position size: 0.25 (25%)
- Target: 75-200 trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14231_6h_ichimoku_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used in signals, but calculated for completeness
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Kumo (cloud) top and bottom
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # TK Cross
    tk_cross_bullish = tenkan > kijun
    tk_cross_bearish = tenkan < kijun
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 52 for Ichimoku, 20 for volume, 14 for ATR, 50 for EMA)
    start = max(52, 20, 14, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine Kumo trend (bullish if Senkou A > Senkou B)
        kumo_bullish = senkou_a[i] > senkou_b[i]
        kumo_bearish = senkou_a[i] < senkou_b[i]
        
        # Check for Kumo twist (change in Kumo trend)
        if i > 0:
            kumo_bullish_prev = senkou_a[i-1] > senkou_b[i-1]
            kumo_bearish_prev = senkou_a[i-1] < senkou_b[i-1]
            kumo_twist_bullish = not kumo_bullish_prev and kumo_bullish  # Bearish to bullish
            kumo_twist_bearish = not kumo_bearish_prev and kumo_bearish  # Bullish to bearish
        else:
            kumo_twist_bullish = False
            kumo_twist_bearish = False
        
        # Ichimoku signals with 1d EMA filter and volume
        # Long: Price > Kumo + TK Cross bullish + Kumo bullish/bullish twist + 1d EMA up + volume
        # Short: Price < Kumo + TK Cross bearish + Kumo bearish/bearish twist + 1d EMA down + volume
        price_above_kumo = close[i] > kumo_top[i]
        price_below_kumo = close[i] < kumo_bottom[i]
        
        ema_rising = ema_1d_aligned[i] > ema_1d_aligned[i-1] if i > 0 else False
        ema_falling = ema_1d_aligned[i] < ema_1d_aligned[i-1] if i > 0 else False
        
        ichimoku_long = (price_above_kumo and 
                        tk_cross_bullish[i] and 
                        (kumo_bullish or kumo_twist_bullish) and
                        ema_rising and 
                        vol_filter[i])
        
        ichimoku_short = (price_below_kumo and 
                         tk_cross_bearish[i] and 
                         (kumo_bearish or kumo_twist_bearish) and
                         ema_falling and 
                         vol_filter[i])
        
        # Generate signals
        if position == 0:
            if ichimoku_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif ichimoku_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price re-enters Kumo or TK Cross turns bearish
            if close[i] <= kumo_top[i] or not tk_cross_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price re-enters Kumo or TK Cross turns bullish
            if close[i] >= kumo_bottom[i] or not tk_cross_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals