#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels from prior week identify key support/resistance; breakouts above R4 or below S4
# with volume confirmation indicate strong momentum. 1w EMA50 ensures trades align with long-term trend
# to avoid false breakouts in choppy markets. Designed for 50-150 total trades over 4 years (12-37/year)
# on 12h timeframe. Works in bull markets (buying breakouts in uptrend) and bear markets
# (selling breakdowns in downtrend) by only taking trades in direction of 1w EMA50.

name = "12h_Camarilla_R4S4_Breakout_1wEMA50_Volume"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate prior week's Camarilla levels (using 1w data)
    # Camarilla: based on prior week's high, low, close
    prior_high = df_1w['high'].shift(1).values  # prior week's high
    prior_low = df_1w['low'].shift(1).values    # prior week's low
    prior_close = df_1w['close'].shift(1).values # prior week's close
    
    # Calculate Camarilla levels (R4/S4 are the most significant breakout levels)
    R4 = prior_close + (prior_high - prior_low) * 1.1 / 2
    S4 = prior_close - (prior_high - prior_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for prior week to complete)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Volume confirmation: 2.0x 20-period average (20*12h = 10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA50 and Camarilla)
    start_idx = max(50, 30)  # 50 bars for EMA50, 30 bars to ensure prior week data available
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R4 with volume spike AND price > 1w EMA50 (bullish trend)
            if (close[i] > R4_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S4 with volume spike AND price < 1w EMA50 (bearish trend)
            elif (close[i] < S4_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below R4 (failed breakout) OR price below 1w EMA50 (trend change)
            if close[i] < R4_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above S4 (failed breakdown) OR price above 1w EMA50 (trend change)
            if close[i] > S4_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals