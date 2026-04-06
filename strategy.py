#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + Tenkan/Kijun Cross with Weekly Slope Filter
# Uses Ichimoku cloud (Senkou Span A/B) as dynamic support/resistance and trend filter.
# Tenkan-Kijun cross provides entry signals, filtered by cloud position and weekly trend.
# Weekly slope (slope of weekly close over 4 periods) ensures we trade with higher timeframe momentum.
# Works in bull/bear because cloud adapts to volatility, and weekly filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost.

name = "exp_13107_6h_ichimoku9_1w_slope_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # Tenkan-sen (Conversion Line)
KIJUN_PERIOD = 26      # Kijun-sen (Base Line)
SENKOU_B_PERIOD = 52   # Senkou Span B
WEEKLY_SLOPE_PERIOD = 4 # Weekly slope lookback
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(KIJUN_PERIOD)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values

def calculate_slope(series, period):
    """Calculate slope of linear regression over period"""
    if len(series) < period:
        return np.nan
    x = np.arange(period)
    y = series[-period:]
    if np.all(np.isnan(y)):
        return np.nan
    # Use polyfit for slope (degree 1)
    try:
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]  # slope
    except:
        return np.nan

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly close for slope
    close_1w = df_1w['close'].values
    weekly_slope = np.full(len(close_1w), np.nan)
    for i in range(WEEKLY_SLOPE_PERIOD-1, len(close_1w)):
        weekly_slope[i] = calculate_slope(close_1w[:i+1], WEEKLY_SLOPE_PERIOD)
    weekly_slope_aligned = align_htf_to_ltf(prices, df_1w, weekly_slope)
    
    # Calculate 6h Ichimoku
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Determine cloud top and bottom
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KIJUN_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly slope not available
        if np.isnan(weekly_slope_aligned[i]):
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
        
        # Ichimoku signals
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top[i] if not np.isnan(cloud_top[i]) else False
        price_below_cloud = close[i] < cloud_bottom[i] if not np.isnan(cloud_bottom[i]) else False
        
        # Tenkan-Kijun cross
        tk_cross_up = (tenkan[i] > kijun[i]) and (i > 0) and (tenkan[i-1] <= kijun[i-1])
        tk_cross_down = (tenkan[i] < kijun[i]) and (i > 0) and (tenkan[i-1] >= kijun[i-1])
        
        # Weekly trend filter
        weekly_uptrend = weekly_slope_aligned[i] > 0
        weekly_downtrend = weekly_slope_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            # Long: price above cloud + TK cross up + weekly uptrend
            if price_above_cloud and tk_cross_up and weekly_uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: price below cloud + TK cross down + weekly downtrend
            elif price_below_cloud and tk_cross_down and weekly_downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below cloud OR TK cross down
            if price_below_cloud or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short: price above cloud OR TK cross up
            if price_above_cloud or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals