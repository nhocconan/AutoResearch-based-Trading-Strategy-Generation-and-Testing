#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and 12h EMA50 trend filter
# Camarilla R3/S3 levels act as magnet points; breakout with volume confirms institutional participation
# 12h EMA50 ensures we only trade in the direction of the intermediate trend
# Volume spike (2x 20-period average) filters low-probability breakouts
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with 12h trend while using 1d for Camarilla calculation

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using typical Camarilla multipliers: 1.1/12, 1.1/6, 1.1/4, 1.1/2
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # R4 = close + 1.1*(high-low)*1.1/2, S4 = close - 1.1*(high-low)*1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = np.roll(prev_close, 1)
    prev_high = np.roll(prev_high, 1)
    prev_low = np.roll(prev_low, 1)
    prev_close[0] = prev_high[0] = prev_low[0] = np.nan  # first bar has no previous
    
    rang = prev_high - prev_low
    r3 = prev_close + 1.1 * rang * (1.1/4)
    s3 = prev_close - 1.1 * rang * (1.1/4)
    r4 = prev_close + 1.1 * rang * (1.1/2)
    s4 = prev_close - 1.1 * rang * (1.1/2)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike (2x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    
    # Align 1d indicators to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Align 12h EMA50 to 6h
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla, EMA50 and volume MA)
    start_idx = 60  # max(20 for volume, 50 for EMA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R3 with volume spike and above 12h EMA50 (uptrend)
            if (close[i] > r3_aligned[i] and 
                volume_spike_aligned[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price < S3 with volume spike and below 12h EMA50 (downtrend)
            elif (close[i] < s3_aligned[i] and 
                  volume_spike_aligned[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price < S3 (mean reversion) OR price > R4 (exhaustion)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price > R3 (mean reversion) OR price < S4 (exhaustion)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals