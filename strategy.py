#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels + volume confirmation
# Fade at R3/S3 levels with volume confirmation, breakout continuation at R4/S4
# Uses discrete position sizing 0.25 to target ~20-40 trades/year
# Works in bull/bear markets: mean reversion at extreme levels, breakout follows trends
# 6h timeframe balances responsiveness with low fee drag

name = "6h_12h_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous day)
    # Camarilla levels: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #                  S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using previous 12h bar for pivot calculation (standard Camarilla)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = np.nan  # First value has no previous
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    
    rangeprev = prev_high_12h - prev_low_12h
    camarilla_r3 = prev_close_12h + 1.1 * rangeprev * 1.1 / 4
    camarilla_s3 = prev_close_12h - 1.1 * rangeprev * 1.1 / 4
    camarilla_r4 = prev_close_12h + 1.1 * rangeprev * 1.1 / 2
    camarilla_s4 = prev_close_12h - 1.1 * rangeprev * 1.1 / 2
    
    # Align 12h Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation: current 6h volume > 1.3x average 6h volume (20-period)
    vol_s = pd.Series(volume)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.3 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for volume MA
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price rises above R4 (breakout continuation) or falls below S3 (mean reversion fail)
            if close[i] > r4_aligned[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price falls below S4 (breakout continuation) or rises above R3 (mean reversion fail)
            if close[i] < s4_aligned[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Fade at extreme levels (R3/S3) with volume confirmation
            if close[i] > r3_aligned[i] and volume_confirmed[i]:
                position = -1  # Short at R3 (expect mean reversion down)
                signals[i] = -0.25
            elif close[i] < s3_aligned[i] and volume_confirmed[i]:
                position = 1   # Long at S3 (expect mean reversion up)
                signals[i] = 0.25
            # Breakout continuation at R4/S4 with volume confirmation
            elif close[i] > r4_aligned[i] and volume_confirmed[i]:
                position = 1   # Long breakout above R4
                signals[i] = 0.25
            elif close[i] < s4_aligned[i] and volume_confirmed[i]:
                position = -1  # Short breakout below S4
                signals[i] = -0.25
    
    return signals