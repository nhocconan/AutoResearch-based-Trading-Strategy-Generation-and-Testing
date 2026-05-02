#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels from prior week identify key support/resistance; breakouts above R3 or below S3
# with volume confirmation indicate strong momentum. 1w EMA34 ensures trades align with weekly trend
# to avoid false breakouts in choppy markets. Designed for 30-100 total trades over 4 years (7-25/year)
# on 1d timeframe. Works in bull markets (buying breakouts in uptrend) and bear markets
# (selling breakdowns in downtrend) by only taking trades in direction of 1w EMA34.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate prior week's Camarilla levels (using 1w data)
    # Camarilla: based on prior week's high, low, close
    prior_high = df_1w['high'].shift(1).values  # prior week's high
    prior_low = df_1w['low'].shift(1).values    # prior week's low
    prior_close = df_1w['close'].shift(1).values # prior week's close
    
    # Calculate Camarilla levels (R3/S3 are the significant breakout levels)
    R3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    S3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (wait for prior week to complete)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Volume confirmation: 2.0x 20-period average (20*1d = ~20 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA34 and Camarilla)
    start_idx = max(34, 30)  # 34 bars for EMA34, 30 bars to ensure prior week data available
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike AND price > 1w EMA34 (bullish trend)
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike AND price < 1w EMA34 (bearish trend)
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below R3 (failed breakout) OR price below 1w EMA34 (trend change)
            if close[i] < R3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (failed breakdown) OR price above 1w EMA34 (trend change)
            if close[i] > S3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals