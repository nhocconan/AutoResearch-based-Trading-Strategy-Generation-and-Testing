#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with volume spike and 1d chop regime filter
# - Long: price breaks above Camarilla H4, volume > 2x 20-bar avg, chop > 61.8 (range) → mean reversion long
# - Short: price breaks below Camarilla L4, volume > 2x 20-bar avg, chop > 61.8 (range) → mean reversion short
# - Exit: price touches Camarilla Pivot Point (PP)
# - Uses 1d chop regime to avoid trending markets where breakouts fail
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within fee drag limits
# - Camarilla H4/L4 are strong reversal levels in ranging markets

name = "4h_1d_camarilla_h4l4_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # PP = (H + L + C) / 3
    # H4 = C + ((H-L) * 1.1/2)
    # L4 = C - ((H-L) * 1.1/2)
    PP = (high_1d + low_1d + close_1d) / 3
    H4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    L4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Pre-compute 1d Chop Index (EHLERS) for regime detection
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Chop Index: higher values = ranging market"""
        tr1 = np.abs(high_arr - low_arr)
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_arr[0] - low_arr[0]
        
        atr = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        # Avoid division by zero
        range_hl = hh - ll
        chop = np.where(range_hl != 0, 100 * np.log10(atr / range_hl) / np.log10(window), 50)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(PP_aligned[i]) or np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        PP_level = PP_aligned[i]
        H4_level = H4_aligned[i]
        L4_level = L4_aligned[i]
        
        # Volume confirmation: current volume > 2x 20-period average (strong participation)
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Regime filter: Chop > 61.8 indicates ranging market (fade breakouts)
        chop_level = chop_1d_aligned[i]
        ranging_market = chop_level > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long setup: price breaks below L4 (false breakdown) in ranging market with volume
        if close_price < L4_level and vol_confirm and ranging_market:
            enter_long = True
        
        # Short setup: price breaks above H4 (false breakout) in ranging market with volume
        if close_price > H4_level and vol_confirm and ranging_market:
            enter_short = True
        
        # Exit conditions: price returns to pivot point (mean reversion complete)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches or exceeds pivot point
            exit_long = close_price >= PP_level
        elif position == -1:
            # Exit short if price reaches or falls below pivot point
            exit_short = close_price <= PP_level
        
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