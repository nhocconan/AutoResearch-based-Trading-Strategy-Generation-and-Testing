#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with daily trend filter and volume confirmation.
# The Ichimoku Cloud (Tenkan-sen/Kijun-sen cross + cloud color) provides clear
# trend direction and support/resistance zones. Combined with daily trend filter
# to avoid counter-trend trades and volume confirmation for institutional
# participation. Works in bull markets (price above cloud, bullish TK cross)
# and bear markets (price below cloud, bearish TK cross).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13335_6h_ichimoku_cloud_daily_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9    # Tenkan-sen period
KIJUN_PERIOD = 26    # Kijun-sen period
SENKOU_B_PERIOD = 52 # Senkou span B period
DMA_PERIOD = 50      # Daily MA for trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past TENKAN_PERIOD
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past KIJUN_PERIOD
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted KIJUN_PERIOD ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(KIJUN_PERIOD)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past SENKOU_B_PERIOD shifted KIJUN_PERIOD ahead
    senkou_span_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                      pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values

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
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily MA for trend filter
    close_1d = df_1d['close'].values
    daily_ma = pd.Series(close_1d).ewm(span=DMA_PERIOD, adjust=False, min_periods=DMA_PERIOD).mean().values
    daily_ma_aligned = align_htf_to_ltf(prices, df_1d, daily_ma)
    
    # Calculate 6h Ichimoku
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b = calculate_ichimoku(high, low, close)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # TK Cross (Tenkan-sen crossing Kijun-sen)
    tk_cross = tenkan_sen - kijun_sen
    tk_cross_above = tk_cross > 0
    tk_cross_below = tk_cross < 0
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, DMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + KIJUN_PERIOD
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(daily_ma_aligned[i]) or np.isnan(volume_ma[i])):
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
        
        # Daily trend filter: price above/below daily MA
        uptrend = close[i] > daily_ma_aligned[i]
        downtrend = close[i] < daily_ma_aligned[i]
        
        # Ichimoku signals
        # Long: Price above cloud + bullish TK cross + uptrend + volume
        long_signal = (price_above_cloud[i] and 
                      tk_cross_above[i] and 
                      uptrend and 
                      volume_ok)
        
        # Short: Price below cloud + bearish TK cross + downtrend + volume
        short_signal = (price_below_cloud[i] and 
                       tk_cross_below[i] and 
                       downtrend and 
                       volume_ok)
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals