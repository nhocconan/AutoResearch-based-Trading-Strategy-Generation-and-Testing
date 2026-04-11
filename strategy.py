#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d: fade at R3/S3 with volume confirmation
# - Long: price touches or slightly breaks S3 with volume spike and closes above S3
# - Short: price touches or slightly breaks R3 with volume spike and closes below R3
# - Exit: price moves to opposite Camarilla level (S3 to R3 or R3 to S3) or reaches R4/S4
# - Uses 1d Camarilla levels calculated from prior 1d OHLC, aligned to 12h
# - Works in ranging markets by fading extremes, avoids strong trends via volume filter
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "12h_1d_camarilla_pivot_fade_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels for current day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict filter)
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Price position relative to Camarilla S3/R3 levels
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        # Entry conditions with hysteresis: require clear rejection of level
        enter_long = False
        enter_short = False
        
        # Long fade: price touches S3 from below with volume and closes above S3
        # Require low touched S3 and close recovered above S3 (bullish rejection)
        if low_price <= s3 and close_price > s3 and vol_confirm:
            enter_long = True
        
        # Short fade: price touches R3 from above with volume and closes below R3
        # Require high touched R3 and close dropped below R3 (bearish rejection)
        if high_price >= r3 and close_price < r3 and vol_confirm:
            enter_short = True
        
        # Exit conditions: mean reversion to opposite level or extreme levels
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches R3 (mean reversion) or breaks S4 (stop)
            s4 = camarilla_s3_aligned[i] - (r3 - s3)  # S4 = S3 - (R3-S3)
            exit_long = close_price >= r3 or close_price <= s4
        elif position == -1:
            # Exit short if price reaches S3 (mean reversion) or breaks R4 (stop)
            r4 = camarilla_r3_aligned[i] + (r3 - s3)  # R4 = R3 + (R3-S3)
            exit_short = close_price <= s3 or close_price >= r4
        
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