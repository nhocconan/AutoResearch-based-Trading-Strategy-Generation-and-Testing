#!/usr/bin/env python3
"""
Experiment #9175: 6h Ichimoku with weekly trend filter + volume confirmation.
Hypothesis: Ichimoku's leading span and Kumo cloud provide dynamic support/resistance, 
weekly trend filter ensures directional alignment, volume confirmation adds conviction.
Works in bull (breakouts above Kumo) and bear (breakdowns below Kumo with trend filter).
Targets 100-200 total trades over 4 years (25-50/year) with controlled risk via ATR stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9175_6h_ichimoku_weekly_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
WEEKLY_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load HTF data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=WEEKLY_EMA_PERIOD, adjust=False, min_periods=WEEKLY_EMA_PERIOD).mean().values
    
    # Price relative to weekly EMA: above = bullish bias, below = bearish bias
    price_vs_weekly_ema = np.where(close_weekly > ema_weekly, 1, 
                                   np.where(close_weekly < ema_weekly, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, price_vs_weekly_ema)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_high = pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max()
    tenkan_low = pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_high = pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max()
    kijun_low = pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(KIJUN_PERIOD)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_high_b = pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max()
    senkou_low_b = pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()
    senkou_b = ((senkou_high_b + senkou_low_b) / 2).shift(KIJUN_PERIOD)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for entry but can be used for confirmation
    
    # Kumo (Cloud) top and bottom
    kumon_top = np.maximum(senkou_a, senkou_b)
    kumon_bottom = np.minimum(senkou_a, senkou_b)
    
    # Convert to numpy arrays and handle NaN
    tenkan_sen = tenkan_sen.values
    kijun_sen = kijun_sen.values
    kumon_top = kumon_top.values
    kumon_bottom = kumon_bottom.values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + KIJUN_PERIOD + 1
    
    for i in range(start, n):
        # Skip if Ichimoku data not available
        if np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(kumon_top[i]) or np.isnan(kumon_bottom[i]):
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
        
        # Determine market bias from weekly EMA
        bull_bias = price_vs_weekly_ema_aligned[i] == 1   # weekly price above EMA20
        bear_bias = price_vs_weekly_ema_aligned[i] == -1  # weekly price below EMA20
        
        # Ichimoku signals
        # Price above Kumo = bullish bias
        price_above_kumo = close[i] > kumon_top[i]
        # Price below Kumo = bearish bias
        price_below_kumo = close[i] < kumon_bottom[i]
        # TK cross (Tenkan crosses above/below Kijun)
        tk_cross_up = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_down = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        # Long: Price above Kumo OR TK cross up with bullish bias + volume
        long_entry = (price_above_kumo and bull_bias and volume_confirmed) or \
                   (tk_cross_up and bull_bias and volume_confirmed and close[i] > kijun_sen[i])
        # Short: Price below Kumo OR TK cross down with bearish bias + volume
        short_entry = (price_below_kumo and bear_bias and volume_confirmed) or \
                    (tk_cross_down and bear_bias and volume_confirmed and close[i] < kijun_sen[i])
        
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