#!/usr/bin/env python3
"""
Experiment #10391: 6h Ichimoku Cloud + Daily Trend + Volume Spike
Hypothesis: Ichimoku cloud breakouts aligned with daily trend (Tenkan-Kijun cross) with volume confirmation
provide high-probability trend continuation. The cloud acts as dynamic support/resistance and filters false breakouts.
Works in bull markets (price above cloud, bullish TK cross) and bear markets (price below cloud, bearish TK cross).
Volume reduces false signals. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, align_htf_to_ltf

name = "exp_10391_6h_ichimoku_cloud_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ichimoku(high, low):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2)
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Set first value to first TR to avoid NaN
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend direction (50-period)
    daily_close = df_daily['close'].values
    daily_ema = calculate_ema(daily_close, 50)
    
    # Align daily EMA to 6h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 6h Ichimoku
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low)
    
    # Calculate cloud top and bottom (account for look-ahead: Senkou spans are plotted 26 periods ahead)
    # For current cloud, we need Senkou A/B from 26 periods ago
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period: need enough data for Ichimoku calculations
    start = max(SENKOU_B_PERIOD + KIJUN_PERIOD, 50) + 1  # 52+26=78, plus daily EMA
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(daily_ema_aligned[i]):
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
        volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        # Ichimoku signals
        # Price above cloud (bullish) or below cloud (bearish)
        price_above_cloud = close[i] > cloud_top[i] if not np.isnan(cloud_top[i]) else False
        price_below_cloud = close[i] < cloud_bottom[i] if not np.isnan(cloud_bottom[i]) else False
        
        # Tenkan-Kijun cross
        tk_bullish_cross = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1]) if i > 0 and not np.isnan(tenkan[i]) and not np.isnan(kijun[i]) else False
        tk_bearish_cross = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1]) if i > 0 and not np.isnan(tenkan[i]) and not np.isnan(kijun[i]) else False
        
        # Entry conditions: aligned with daily trend, price vs cloud, TK cross, volume
        long_entry = price_above_cloud and above_daily_ema and tk_bullish_cross and volume_spike
        short_entry = price_below_cloud and below_daily_ema and tk_bearish_cross and volume_spike
        
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