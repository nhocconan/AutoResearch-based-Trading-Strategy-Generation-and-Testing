#!/usr/bin/env python3
"""
exp_7307_6h_ichimoku_tk_cross_1d_cloud_v1
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter. 
In bull regime (price > 1d Senkou Span A/B): long on TK cross above cloud, short on cross below.
In bear regime (price < 1d cloud): short on TK cross below cloud, long on cross above.
Uses volume confirmation to avoid whipsaws. Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to 1d cloud-defined regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7307_6h_ichimoku_tk_cross_1d_cloud_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ICHI_CONVERSION = 9   # Tenkan-sen
ICHI_BASE = 26        # Kijun-sen
ICHI_LEADING_B = 52   # Senkou Span B
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8     # ~32 hours (more conservative for 6h)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over last 9 periods
    tenkan_1d = (pd.Series(high_1d).rolling(window=ICHI_CONVERSION, min_periods=ICHI_CONVERSION).max() + 
                 pd.Series(low_1d).rolling(window=ICHI_CONVERSION, min_periods=ICHI_CONVERSION).min()) / 2
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over last 26 periods
    kijun_1d = (pd.Series(high_1d).rolling(window=ICHI_BASE, min_periods=ICHI_BASE).max() + 
                pd.Series(low_1d).rolling(window=ICHI_BASE, min_periods=ICHI_BASE).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(ICHI_BASE)
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over last 52 periods shifted 26 ahead
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=ICHI_LEADING_B, min_periods=ICHI_LEADING_B).max() + 
                    pd.Series(low_1d).rolling(window=ICHI_LEADING_B, min_periods=ICHI_LEADING_B).min()) / 2).shift(ICHI_BASE)
    
    # Align to LTF (6h)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
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
    start = max(ICHI_BASE, ICHI_LEADING_B, VOL_MA_PERIOD, ATR_PERIOD) + ICHI_BASE + 10
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or \
           np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]):
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
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Determine market regime based on price vs cloud
        above_cloud = close[i] > upper_cloud
        below_cloud = close[i] < lower_cloud
        in_cloud = (close[i] >= lower_cloud) & (close[i] <= upper_cloud)
        
        # TK cross signals
        tk_cross_above = tenkan_1d_aligned[i] > kijun_1d_aligned[i] and tenkan_1d_aligned[i-1] <= kijun_1d_aligned[i-1]
        tk_cross_below = tenkan_1d_aligned[i] < kijun_1d_aligned[i] and tenkan_1d_aligned[i-1] >= kijun_1d_aligned[i-1]
        
        # Long signals: TK cross above in bull regime OR TK cross above from below cloud in bear regime
        long_signal = False
        if above_cloud and tk_cross_above and vol_confirmed:
            long_signal = True  # Continuation in bull
        elif below_cloud and tk_cross_above and close[i] > lower_cloud and vol_confirmed:
            long_signal = True  # Breakout above cloud from below
        
        # Short signals: TK cross below in bear regime OR TK cross below from above cloud in bull regime
        short_signal = False
        if below_cloud and tk_cross_below and vol_confirmed:
            short_signal = True  # Continuation in bear
        elif above_cloud and tk_cross_below and close[i] < upper_cloud and vol_confirmed:
            short_signal = True  # Breakdown below cloud from above
        
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