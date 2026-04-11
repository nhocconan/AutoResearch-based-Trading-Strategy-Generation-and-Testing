#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_camarilla_pivot_bounce_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 500:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return signals
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla pivot calculation using previous day's OHLC
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Resistance levels
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    r2_12h = close_12h + (range_12h * 1.1 / 6)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    r4_12h = close_12h + (range_12h * 1.1 / 2)
    
    # Support levels
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    s2_12h = close_12h - (range_12h * 1.1 / 6)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    s4_12h = close_12h - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate 1d trend direction using EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average on 6h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(500, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price bounces off S3 with 1d uptrend + volume confirmation
        price_near_s3 = price_close <= s3_12h_aligned[i] * 1.005  # Within 0.5% of S3
        price_above_s4 = price_close > s4_12h_aligned[i]
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        if price_near_s3 and uptrend_1d and vol_confirm:
            enter_long = True
        
        # Short: Price bounces off R3 with 1d downtrend + volume confirmation
        price_near_r3 = price_close >= r3_12h_aligned[i] * 0.995  # Within 0.5% of R3
        price_below_r4 = price_close < r4_12h_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        if price_near_r3 and downtrend_1d and vol_confirm:
            enter_short = True
        
        # Exit conditions: price reaches opposite S4/R4 level
        exit_long = price_close >= r4_12h_aligned[i]
        exit_short = price_close <= s4_12h_aligned[i]
        
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

# Hypothesis: Camarilla pivot bounce strategy on 6h timeframe with 12h S3/R3 bounce levels and 1d trend filter.
# Enters long near S3 in uptrend, short near R3 in downtrend with volume confirmation.
# Exits at opposite S4/R4 levels for defined risk-reward.
# Works in both bull and bear markets by following 1d trend while fading extreme intraday moves.
# Target: 50-150 total trades over 4 years with position size 0.25 to manage risk.