#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud system with 1w trend filter. Uses TK cross + cloud twist for entries, 
# weekly cloud color for trend filter. Works in bull/bear because Ichimoku adapts to volatility 
# and cloud acts as dynamic support/resistance. Weekly filter ensures we only trade with major trend.
# Target: 80-120 trades over 4 years (20-30/year) to balance opportunity and cost.

name = "ichimoku_6h_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
CHIKOU_SHIFT = 26
WEEKLY_TREND_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods back
    chikou_span = pd.Series(close)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values, chikou_span.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, WEEKLY_TREND_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h Ichimoku
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # TK cross signals (with hysteresis to prevent whipsaw)
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    price_in_cloud = (close >= cloud_bottom) & (close <= cloud_top)
    
    # ATR for stops
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KIJUN_PERIOD, SENKOU_B_PERIOD, WEEKLY_TREND_PERIOD, ATR_PERIOD) + CHIKOU_SHIFT + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # Weekly trend filter: only trade with weekly trend
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        # Ichimoku signals with cloud filter
        # Long: TK cross up + price above cloud + weekly uptrend
        long_signal = tk_cross_up[i] and price_above_cloud[i] and weekly_uptrend
        # Short: TK cross down + price below cloud + weekly downtrend
        short_signal = tk_cross_down[i] and price_below_cloud[i] and weekly_downtrend
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Stay long if still above cloud bottom or in cloud with bullish bias
            if close[i] > cloud_bottom[i] or (price_in_cloud[i] and tenkan[i] > kijun[i]):
                signals[i] = SIGNAL_SIZE
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Stay short if still below cloud top or in cloud with bearish bias
            if close[i] < cloud_top[i] or (price_in_cloud[i] and tenkan[i] < kijun[i]):
                signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = 0.0
                position = 0
    
    return signals