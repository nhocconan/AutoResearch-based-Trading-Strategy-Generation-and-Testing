#!/usr/bin/env python3
"""
exp_7027_6h_ichimoku_1d_cloud_v1
Hypothesis: 6h Ichimoku strategy with 1d cloud filter for trend direction.
In bull markets (price > 1d Senkou Span A/B): long on TK cross above cloud.
In bear markets (price < 1d Senkou Span A/B): short on TK cross below cloud.
Volume confirmation filters false signals. Designed for 6h timeframe to capture swings with ~12-37 trades/year.
Works in both bull and bear markets by aligning with 1d cloud trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7027_6h_ichimoku_1d_cloud_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9
KJ_PERIOD = 26
SSB_PERIOD = 52
DISPLACEMENT = 26
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 40  # ~10 months (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (period high + period low) / 2
    period_high_tk = pd.Series(high_1d).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max().values
    period_low_tk = pd.Series(low_1d).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min().values
    tenkan_sen = (period_high_tk + period_low_tk) / 2
    
    # Kijun-sen (Base Line): (period high + period low) / 2
    period_high_kj = pd.Series(high_1d).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max().values
    period_low_kj = pd.Series(low_1d).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min().values
    kijun_sen = (period_high_kj + period_low_kj) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (period high + period low) / 2
    period_high_ssb = pd.Series(high_1d).rolling(window=SSB_PERIOD, min_periods=SSB_PERIOD).max().values
    period_low_ssb = pd.Series(low_1d).rolling(window=SSB_PERIOD, min_periods=SSB_PERIOD).min().values
    senkou_span_b = (period_high_ssb + period_low_ssb) / 2
    
    # Align Ichimoku components to LTF (6h) with shift(1) for completed bars only
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(KJ_PERIOD, SSB_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + DISPLACEMENT + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
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
        vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine cloud boundaries and trend
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # TK Cross signals
        tk_cross_above = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_below = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Ichimoku signals aligned with cloud trend
        long_signal = price_above_cloud and tk_cross_above and vol_confirmed
        short_signal = price_below_cloud and tk_cross_below and vol_confirmed
        
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

}