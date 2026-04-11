#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop regime filter
# - Long: price breaks above Camarilla H3 level, volume > 2x 20-period avg, CHOP(14) > 61.8 (ranging)
# - Short: price breaks below Camarilla L3 level, volume > 2x 20-period avg, CHOP(14) > 61.8 (ranging)
# - Exit: price returns to Camarilla Pivot point or opposite H3/L3 level
# - Uses 1d EMA(50) trend filter for bias: price > EMA(50) for long bias, price < EMA(50) for short bias
# - Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag
# - Works in ranging markets by fading extremes at Camarilla levels with volume confirmation

name = "4h_1d_camarilla_breakout_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend bias
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day OHLC)
    # Camarilla levels: H4, H3, H2, H1, Pivot, L1, L2, L3, L4
    # We use H3/L3 for entries and Pivot for exits
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high, 1)  # Note: this is approximate, but 1d data is daily
    prev_low_1d = np.roll(low, 1)
    
    # Fix first values
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high[0]
    prev_low_1d[0] = low[0]
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    h3_1d = pivot_1d + range_1d * 1.1 / 4
    l3_1d = pivot_1d - range_1d * 1.1 / 4
    
    # Align 1d Camarilla levels to 4h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum_14 / (np.log10(14) * (max_high_14 - min_low_14)))
    # Handle division by zero or invalid values
    chop = np.where((max_high_14 - min_low_14) > 0, chop, 50.0)  # default to neutral
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(chop[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = h3_1d_aligned[i]
        l3 = l3_1d_aligned[i]
        pivot = pivot_1d_aligned[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Regime filter: CHOP > 61.8 (ranging market)
        ranging = chop[i] > 61.8
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above H3, volume confirmation, ranging market, long bias
        if close_price > h3 and vol_confirm and ranging and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below L3, volume confirmation, ranging market, short bias
        if close_price < l3 and vol_confirm and ranging and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Pivot or drops below L3 (stop)
            exit_long = close_price <= pivot
        elif position == -1:
            # Exit short if price returns to Pivot or rises above H3 (stop)
            exit_short = close_price >= pivot
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals