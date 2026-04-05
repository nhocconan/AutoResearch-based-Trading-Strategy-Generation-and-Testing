#!/usr/bin/env python3
"""
exp_6979_6h_ichimoku_cloud_1d_adx_v1
Hypothesis: 6h Ichimoku cloud breakout with 1d ADX trend filter and volume confirmation.
In strong uptrends (1d ADX>25): long when price breaks above Kumo cloud and TK cross bullish.
In strong downtrends (1d ADX>25): short when price breaks below Kumo cloud and TK cross bearish.
In weak trends (1d ADX<=25): fade at cloud edges (mean reversion).
Ichimoku provides dynamic support/resistance, ADX filters for trend strength, volume confirms breakouts.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by aligning with 1d trend regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6979_6h_ichimoku_cloud_1d_adx_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ICHI_CONVERSION_PERIOD = 9   # Tenkan-sen
ICHI_BASE_PERIOD = 26        # Kijun-sen
ICHI_LEADING_SPAN_B_PERIOD = 52  # Senkou Span B
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 20  # ~5 months (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for ADX and Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over Conversion Period
    highest_high_9 = pd.Series(high_1d).rolling(window=ICHI_CONVERSION_PERIOD, min_periods=ICHI_CONVERSION_PERIOD).max().values
    lowest_low_9 = pd.Series(low_1d).rolling(window=ICHI_CONVERSION_PERIOD, min_periods=ICHI_CONVERSION_PERIOD).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over Base Period
    highest_high_26 = pd.Series(high_1d).rolling(window=ICHI_BASE_PERIOD, min_periods=ICHI_BASE_PERIOD).max().values
    lowest_low_26 = pd.Series(low_1d).rolling(window=ICHI_BASE_PERIOD, min_periods=ICHI_BASE_PERIOD).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over Leading Span B Period shifted forward 26 periods
    highest_high_52 = pd.Series(high_1d).rolling(window=ICHI_LEADING_SPAN_B_PERIOD, min_periods=ICHI_LEADING_SPAN_B_PERIOD).max().values
    lowest_low_52 = pd.Series(low_1d).rolling(window=ICHI_LEADING_SPAN_B_PERIOD, min_periods=ICHI_LEADING_SPAN_B_PERIOD).min().values
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Calculate 1d ADX
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (EMA with alpha=1/period)
    atr_1d = tr.ewm(alpha=1/ADX_PERIOD, adjust=False).mean().values
    dm_plus_smooth = dm_plus.ewm(alpha=1/ADX_PERIOD, adjust=False).mean().values
    dm_minus_smooth = dm_minus.ewm(alpha=1/ADX_PERIOD, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/ADX_PERIOD, adjust=False).mean().values
    
    # Align HTF indicators to LTF (6h)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(
        ICHI_BASE_PERIOD, ICHI_LEADING_SPAN_B_PERIOD,  # Ichimoku needs base periods
        ADX_PERIOD * 2,  # ADX needs extra smoothing
        VOL_MA_PERIOD, ATR_PERIOD
    ) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]) or np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]):
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
        
        # Determine Kumo (cloud) boundaries
        senkou_span_a_val = senkou_span_a_aligned[i]
        senkou_span_b_val = senkou_span_b_aligned[i]
        upper_kumo = max(senkou_span_a_val, senkou_span_b_val)
        lower_kumo = min(senkou_span_a_val, senkou_span_b_val)
        
        # TK Cross (Tenkan-sen / Kijun-sen cross)
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Determine trend regime from 1d ADX
        strong_trend = adx_aligned[i] > ADX_THRESHOLD
        weak_trend = adx_aligned[i] <= ADX_THRESHOLD
        
        # Initialize signal
        new_signal = 0.0
        
        if strong_trend:
            # Strong trend: follow Ichimoku breakouts with TK cross
            # Long: price above cloud + bullish TK cross + volume
            long_breakout = (close[i] > upper_kumo) and tk_cross_bullish and vol_confirmed
            # Short: price below cloud + bearish TK cross + volume
            short_breakout = (close[i] < lower_kumo) and tk_cross_bearish and vol_confirmed
            
            if long_breakout and position <= 0:
                new_signal = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout and position >= 0:
                new_signal = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                # Hold current position
                new_signal = position * SIGNAL_SIZE if position != 0 else 0.0
                
        else:
            # Weak trend: mean reversion at cloud edges
            # Long: price touches lower cloud + oversold bounce
            long_reversion = (close[i] <= lower_kumo * 1.005) and (tenkan_sen_aligned[i] > kijun_sen_aligned[i])
            # Short: price touches upper cloud + overbought rejection
            short_reversion = (close[i] >= upper_kumo * 0.995) and (tenkan_sen_aligned[i] < kijun_sen_aligned[i])
            
            if long_reversion and position <= 0:
                new_signal = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_reversion and position >= 0:
                new_signal = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                # Hold current position
                new_signal = position * SIGNAL_SIZE if position != 0 else 0.0
        
        signals[i] = new_signal
    
    return signals