#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + chop regime filter
# - Long when price touches Camarilla L3 support with volume spike in choppy market (CHOP > 61.8)
# - Short when price touches Camarilla H3 resistance with volume spike in choppy market
# - Exit when price reaches Camarilla H4/L4 or volume drops
# - Uses 1d Camarilla levels (proven edge from top performers) with 12h execution for better timing
# - Chop regime filter avoids trending markets where pivot fails
# - Volume confirmation ensures institutional participation
# - Target: 12-30 trades/year to stay within fee drag limits

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 12h data ONCE before loop for volume and chop (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    # Range = high - low
    # L3 = close - (range * 1.1/4)
    # L4 = close - (range * 1.1/2)
    # H3 = close + (range * 1.1/4)
    # H4 = close + (range * 1.1/2)
    daily_range = high_1d - low_1d
    camarilla_l3 = close_1d - (daily_range * 1.1 / 4)
    camarilla_l4 = close_1d - (daily_range * 1.1 / 2)
    camarilla_h3 = close_1d + (daily_range * 1.1 / 4)
    camarilla_h4 = close_1d + (daily_range * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    
    # Pre-compute 12h volume SMA for confirmation
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute 12h Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_12h = np.maximum(
        df_12h['high'].values - df_12h['low'].values,
        np.maximum(
            np.abs(df_12h['high'].values - np.roll(df_12h['close'].values, 1)),
            np.abs(df_12h['low'].values - np.roll(df_12h['close'].values, 1))
        )
    )
    # Handle first bar
    tr_12h[0] = df_12h['high'].values[0] - df_12h['low'].values[0]
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(df_12h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_12h['low'].values).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index calculation
    chop_denominator = highest_high_14 - lowest_low_14
    chop_denominator = np.where(chop_denominator == 0, 1, chop_denominator)  # Avoid div by zero
    chop_numerator = atr_14_12h * 14
    chop = 100 * np.log10(chop_numerator / chop_denominator) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # Default to neutral if invalid
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Align 12h volume for current bar check
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    for i in range(30, n):  # Start after 30-bar warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        vol_confirm = volume_12h_aligned[i] > 1.5 * volume_sma_20_12h_aligned[i]
        
        # Chop regime filter: CHOP > 61.8 = ranging market (good for pivot reversals)
        chop_regime = chop_aligned[i] > 61.8
        
        # Price levels
        price = close[i]
        l3 = camarilla_l3_aligned[i]
        l4 = camarilla_l4_aligned[i]
        h3 = camarilla_h3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        
        # Entry conditions with tolerance for touch (0.1% of price)
        tolerance = price * 0.001
        touch_l3 = abs(price - l3) <= tolerance
        touch_h3 = abs(price - h3) <= tolerance
        
        # Exit conditions: reach opposite H/L level or volume drops
        exit_long = price >= h4 or not vol_confirm
        exit_short = price <= l4 or not vol_confirm
        
        # Trading logic
        if touch_l3 and vol_confirm and chop_regime and position != 1:
            position = 1
            signals[i] = 0.25
        elif touch_h3 and vol_confirm and chop_regime and position != -1:
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