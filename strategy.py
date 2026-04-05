#!/usr/bin/env python3
"""
exp_7447_6h_ichimoku_1d_cloud_filter_v1
Hypothesis: 6h Ichimoku with 1d cloud filter for trend direction and weekly volume confirmation.
Ichimoku TK cross on 6h timeframe provides timely entries, while 1d cloud (Senkou Span A/B) filters for strong trend alignment.
Weekly volume surge confirms institutional participation. Designed to work in bull/bear markets by following the higher timeframe trend.
Target: 50-150 trades over 4 years (12-37/year) with 0.25 position size.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7447_6h_ichimoku_1d_cloud_filter_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9      # Tenkan-sen period
KJ_PERIOD = 26     # Kijun-sen period
SS_PERIOD = 52     # Senkou Span period
VOL_MA_PERIOD = 20 # Volume moving average
VOL_SURGE_THRESHOLD = 2.5  # Volume surge multiplier
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 4  # Max 4 bars (~1 day) for 6h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')   # Daily for cloud filter
    df_1w = get_htf_data(prices, '1w')   # Weekly for volume confirmation
    
    # Calculate Ichimoku components on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tk_high = pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max()
    tk_low = pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()
    tenkan_sen = (tk_high + tk_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kj_high = pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max()
    kj_low = pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()
    kijun_sen = (kj_high + kj_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    ss_high = pd.Series(high).rolling(window=SS_PERIOD, min_periods=SS_PERIOD).max()
    ss_low = pd.Series(low).rolling(window=SS_PERIOD, min_periods=SS_PERIOD).min()
    senkou_b = ((ss_high + ss_low) / 2)
    
    # Align Ichimoku components to current 6h bars (no shift needed as they're already calculated)
    tenkan_sen_vals = tenkan_sen.values
    kijun_sen_vals = kijun_sen.values
    senkou_a_vals = senkou_a.values
    senkou_b_vals = senkou_b.values
    
    # Calculate 1d cloud (Kumo) - Senkou Span A/B from 1d data
    # For cloud, we need the actual Senkou Span values from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku components
    tk_high_1d = pd.Series(high_1d).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max()
    tk_low_1d = pd.Series(low_1d).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()
    tenkan_sen_1d = (tk_high_1d + tk_low_1d) / 2
    
    kj_high_1d = pd.Series(high_1d).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max()
    kj_low_1d = pd.Series(low_1d).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()
    kijun_sen_1d = (kj_high_1d + kj_low_1d) / 2
    
    ss_high_1d = pd.Series(high_1d).rolling(window=SS_PERIOD, min_periods=SS_PERIOD).max()
    ss_low_1d = pd.Series(low_1d).rolling(window=SS_PERIOD, min_periods=SS_PERIOD).min()
    senkou_a_1d = ((tk_high_1d + tk_low_1d) / 2 + (kj_high_1d + kj_low_1d) / 2) / 2
    senkou_b_1d = ((ss_high_1d + ss_low_1d) / 2 + (ss_high_1d + ss_low_1d) / 2) / 2
    
    # Align 1d cloud to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # Weekly volume for confirmation
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Current 6h volume
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
    
    # Start from warmup period - need enough data for Ichimoku calculations
    start = max(KJ_PERIOD, SS_PERIOD, TK_PERIOD) + KJ_PERIOD  # +26 for Senkou Span shift
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_vals[i]) or np.isnan(kijun_sen_vals[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_ma_1w_aligned[i])):
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
            
        # Ichimoku signals
        tk_cross_above = tenkan_sen_vals[i] > kijun_sen_vals[i] and tenkan_sen_vals[i-1] <= kijun_sen_vals[i-1]
        tk_cross_below = tenkan_sen_vals[i] < kijun_sen_vals[i] and tenkan_sen_vals[i-1] >= kijun_sen_vals[i-1]
        
        # Cloud filter: price above/both Senkou lines = bullish, below/both = bearish
        cloud_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation: current 6h volume > weekly average * threshold
        vol_surge = volume[i] > vol_ma_1w_aligned[i] * VOL_SURGE_THRESHOLD if not np.isnan(vol_ma_1w_aligned[i]) else False
        
        # Entry logic: TK cross in direction of cloud with volume surge
        if position == 0:
            # Long: TK cross above + price above cloud + volume surge
            if tk_cross_above and price_above_cloud and vol_surge:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Short: TK cross below + price below cloud + volume surge
            elif tk_cross_below and price_below_cloud and vol_surge:
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