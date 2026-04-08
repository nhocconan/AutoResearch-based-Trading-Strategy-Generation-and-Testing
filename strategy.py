#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_1w_regime
# Hypothesis: Camarilla pivot levels from weekly data provide strong support/resistance on 12h timeframe.
# Long when price touches S3 level with bullish 1w trend and volume spike.
# Short when price touches R3 level with bearish 1w trend and volume spike.
# Uses weekly trend filter to avoid counter-trend trades and volume confirmation to ensure momentum.
# Target: 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_1w_regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (using previous week's data)
    # Camarilla formulas: 
    # H4 = close + 1.1*(high-low)*1.1/2
    # L4 = close - 1.1*(high-low)*1.1/2
    # We use S3/L3 and H3/R3 which are closer to the price
    # S3 = close - 1.1*(high-low)*1.1/4
    # R3 = close + 1.1*(high-low)*1.1/4
    # Actually standard Camarilla:
    # S1 = close - (high-low)*1.1/12
    # S2 = close - (high-low)*1.1/6
    # S3 = close - (high-low)*1.1/4
    # R3 = close + (high-low)*1.1/4
    # R2 = close + (high-low)*1.1/6
    # R1 = close + (high-low)*1.1/12
    
    # Calculate for previous week (shift by 1 to avoid look-ahead)
    if len(high_1w) < 2:
        return np.zeros(n)
        
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan  # First value has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Align to 12h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    
    # Weekly trend filter: EMA20 on weekly close
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: 24-period average on 12h (2 days)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or weekly trend turns bearish
            if (close[i] < camarilla_s3_aligned[i]) or (close[i] < ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or weekly trend turns bullish
            if (close[i] > camarilla_r3_aligned[i]) or (close[i] > ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.8x average volume (higher threshold for fewer trades)
            volume_ok = volume[i] > 1.8 * avg_volume[i]
            
            # Long entry: price touches S3 level with bullish weekly trend and volume spike
            if (abs(close[i] - camarilla_s3_aligned[i]) < 0.005 * camarilla_s3_aligned[i]) and \
               (close[i] > ema_20_1w_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R3 level with bearish weekly trend and volume spike
            elif (abs(close[i] - camarilla_r3_aligned[i]) < 0.005 * camarilla_r3_aligned[i]) and \
                 (close[i] < ema_20_1w_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals