#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d: long when price breaks above R3 with volume confirmation and closes in upper half, short when breaks below S3 with volume and closes in lower half. Exit at R4/S4 (extended levels) or opposite Camarilla level (R3/S3). Uses 1d OHLC from prior day, aligned to 12h. Works in bull/bear by capturing momentum at extreme levels while avoiding false breakouts via volume and price close filters. Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits.

name = "12h_1d_camarilla_breakout_v1"
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
    # R4 = close + 1.1*(high-low)*1.1/2
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # S4 = close - 1.1*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_s4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels for current day)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Price position relative to Camarilla levels
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # 12h bar close position (upper/lower half)
        bar_range = high_price - low_price
        if bar_range > 0:
            close_position = (close_price - low_price) / bar_range  # 0=low, 1=high
        else:
            close_position = 0.5
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above R3 with volume and closes in upper half
        if close_price > r3 and vol_confirm and close_position > 0.5:
            enter_long = True
        
        # Short breakout: price breaks below S3 with volume and closes in lower half
        if close_price < s3 and vol_confirm and close_position < 0.5:
            enter_short = True
        
        # Exit conditions: at extended levels (R4/S4) or opposite Camarilla level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches R4 or drops back to S3
            exit_long = close_price >= r4 or close_price <= s3
        elif position == -1:
            # Exit short if price reaches S4 or rises back to R3
            exit_short = close_price <= s4 or close_price >= r3
        
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