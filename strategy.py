#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and 1d volume spike
# Camarilla levels provide intraday support/resistance; breakouts above R3 or below S3 indicate strong momentum
# 4h EMA50 filter ensures we only trade in direction of higher timeframe trend
# 1d volume spike (2x 20-period average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Discrete position sizing 0.20 limits drawdown and minimizes fee churn
# Targets 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# Works in bull markets via breakout continuation and bear markets via fade of false breaks

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dVolSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data ONCE before loop for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume spike (2x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().shift(1).values
    vol_spike_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate Camarilla levels for previous day (using 1d OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # We use previous day's OHLC to calculate today's levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    R3 = prev_close + 1.125 * (prev_high - prev_low)
    S3 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA50, volume MA, and Camarilla)
    start_idx = 60  # max(50 for EMA, 20 for volume MA) + buffer
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, 4h uptrend, and volume spike
            if (close[i] > R3_aligned[i] and 
                close[i] > ema50_4h_aligned[i] and  # price above 4h EMA50 (uptrend)
                vol_spike_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3, 4h downtrend, and volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema50_4h_aligned[i] and  # price below 4h EMA50 (downtrend)
                  vol_spike_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below 4h EMA50 or reverses below R3
            if close[i] < ema50_4h_aligned[i] or close[i] < R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA50 or reverses above S3
            if close[i] > ema50_4h_aligned[i] or close[i] > S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals