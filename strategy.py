#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h trend filter and volume spike
# Camarilla levels from 12h: R3/S3 = strong intraday support/resistance, R4/S4 = breakout levels
# 12h EMA50 trend filter ensures we trade with higher timeframe momentum
# Volume spike (2.0x 20-period average) confirms participation
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by using 12h trend filter to avoid counter-trend trades
# Uses 12h for HTF regime and Camarilla calculation for stability

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Use previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(df_12h['high'].values, 1)
    prev_low = np.roll(df_12h['low'].values, 1)
    prev_close = np.roll(df_12h['close'].values, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = 0  # first bar
    
    # Calculate Camarilla levels
    camarilla_factor = 1.1 * (prev_high - prev_low) * 1.1
    r4 = prev_close + camarilla_factor / 2
    r3 = prev_close + camarilla_factor / 4
    s3 = prev_close - camarilla_factor / 4
    s4 = prev_close - camarilla_factor / 2
    
    # Align 12h indicators to 6h
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Calculate 6h volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA50 and volume MA)
    start_idx = 60  # max(20 for volume, 50 for EMA50) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 with volume in uptrend, or breaks above R4 in any trend
            if ((close[i] > r3_aligned[i] and uptrend and volume_confirm[i]) or
                (close[i] > r4_aligned[i] and volume_confirm[i])):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume in downtrend, or breaks below S4 in any trend
            elif ((close[i] < s3_aligned[i] and downtrend and volume_confirm[i]) or
                  (close[i] < s4_aligned[i] and volume_confirm[i])):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price falls below S3 (in uptrend) or S4 (any trend)
            if (close[i] < s3_aligned[i] and uptrend) or (close[i] < s4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price rises above R3 (in downtrend) or R4 (any trend)
            if (close[i] > r3_aligned[i] and downtrend) or (close[i] > r4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals