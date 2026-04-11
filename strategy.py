#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 1d trend filter
# - Long: price breaks above Camarilla H3 level, volume > 2.0x 20-period avg, price > 1d EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 2.0x 20-period avg, price < 1d EMA(50)
# - Exit: price returns to Camarilla pivot point (P)
# - Uses discrete position sizing (0.25) to limit fee drag
# - Target: 20-35 trades/year (80-140 total over 4 years) to stay within fee drag limits
# - Camarilla levels provide institutional support/resistance that works in both bull and bear markets
# - Volume confirmation ensures breakout validity
# - 1d EMA filter provides higher timeframe trend bias

name = "4h_1d_camarilla_pivot_breakout_v1"
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
    
    # Load 1d data ONCE before loop for pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d Camarilla pivot levels (using previous day's OHLC)
    # Camarilla formulas: P = (H+L+C)/3, Range = H-L
    # H4 = P + 1.1*Range/2, H3 = P + 1.1*Range/4, H2 = P + 1.1*Range/6, H1 = P + 1.1*Range/12
    # L1 = P - 1.1*Range/12, L2 = P - 1.1*Range/6, L3 = P - 1.1*Range/4, L4 = P - 1.1*Range/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels for each 1d bar (using previous day's data to avoid look-ahead)
    pivot = np.full(len(close_1d), np.nan)
    h3 = np.full(len(close_1d), np.nan)
    l3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):  # Start from 1 to use previous day's data
        # Use previous day's OHLC to avoid look-ahead
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Camarilla calculations
        p = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        pivot[i] = p
        h3[i] = p + 1.1 * range_val / 4.0
        l3[i] = p - 1.1 * range_val / 4.0
    
    # Align Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(volume_sma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        pivot_level = pivot_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # 1d EMA trend filter
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above H3, volume confirmation, long bias
        if close_price > h3_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below L3, volume confirmation, short bias
        if close_price < l3_level and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point
            exit_long = close_price <= pivot_level
        elif position == -1:
            # Exit short if price returns to pivot point
            exit_short = close_price >= pivot_level
        
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