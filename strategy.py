#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with volume spike and chop regime filter
# - Long: price crosses above Camarilla H3 (1d) with volume > 2x 20-period average and CHOP(14) > 61.8 (ranging market)
# - Short: price crosses below Camarilla L3 (1d) with volume > 2x 20-period average and CHOP(14) > 61.8
# - Exit: price returns to Camarilla H4/L4 levels or opposite pivot touch
# - Uses discrete position sizing (0.25) to limit fee drag
# - Target: 12-25 trades/year (50-100 total over 4 years) to stay within fee limits for 12h timeframe
# - Works in ranging markets by fading extreme moves with volume confirmation

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots and chop regime (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4, H3, L3, L4
    camarilla_h4 = close_1d + range_1d * 1.1 / 2
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    camarilla_l4 = close_1d - range_1d * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (CHOP) regime filter on 12h data
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_12h[0] = high[0] - low[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_12h * 14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (fade extreme moves)
        ranging_market = chop[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price crosses above H3 with volume confirmation in ranging market
        if close_price > h3 and close_price <= high[i] and vol_confirm and ranging_market:
            enter_long = True
        
        # Short: price crosses below L3 with volume confirmation in ranging market
        if close_price < l3 and close_price >= low[i] and vol_confirm and ranging_market:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to H4 or touches L3 (mean reversion complete)
            exit_long = close_price >= h4 or close_price <= l3
        elif position == -1:
            # Exit short if price returns to L4 or touches H3 (mean reversion complete)
            exit_short = close_price <= l4 or close_price >= h3
        
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