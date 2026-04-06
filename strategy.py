#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku cloud filter with 10-period TK cross and 1d cloud color filter.
# Uses Ichimoku as a trend filter (price above/below cloud) and TK cross for entry timing.
# Weekly trend filter ensures we trade with higher timeframe momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost.

name = "exp_13095_6h_ichimoku10_1w_cloud_tk_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9
KJ_PERIOD = 26
SENB_B_PERIOD = 52
WEEKLY_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over TK_PERIOD
    tenkan_sen = (pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
                  pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over KJ_PERIOD
    kijun_sen = (pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
                 pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward KJ_PERIOD
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(KJ_PERIOD)
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over SENB_B_PERIOD shifted forward KJ_PERIOD
    senkou_span_b = ((pd.Series(high).rolling(window=SENB_B_PERIOD, min_periods=SENB_B_PERIOD).max() + 
                      pd.Series(low).rolling(window=SENB_B_PERIOD, min_periods=SENB_B_PERIOD).min()) / 2).shift(KJ_PERIOD)
    # Chikou Span (Lagging Span): Close shifted back KJ_PERIOD
    chikou_span = pd.Series(close).shift(-KJ_PERIOD)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

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
    volume = prices['volume'].values
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Price above/below cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # TK cross signals
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ok = volume > (volume_ma * VOLUME_THRESHOLD)
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TK_PERIOD, KJ_PERIOD, SENB_B_PERIOD, WEEKLY_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + KJ_PERIOD + 1
    
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
        
        # Entry conditions
        if position == 0:
            # Long: price above cloud, TK cross up, weekly uptrend, volume
            long_signal = price_above_cloud[i] and tk_cross_up[i] and close[i] > ema_1w_aligned[i] and volume_ok[i]
            # Short: price below cloud, TK cross down, weekly downtrend, volume
            short_signal = price_below_cloud[i] and tk_cross_down[i] and close[i] < ema_1w_aligned[i] and volume_ok[i]
            
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals