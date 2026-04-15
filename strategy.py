#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation
# Works in bull: breaks above R3 in uptrend with volume
# Works in bear: breaks below S3 in downtrend with volume
# Uses discrete position sizing (0.25) to minimize fee drag
# Target: 12-37 trades/year (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1w HTF data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of previous day)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use previous day's HLC to avoid look-ahead
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    rng = h_1d - l_1d
    r3 = c_1d + rng * 1.1 / 4
    s3 = c_1d - rng * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1d volume SMA(20) for volume confirmation
    vol_sma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x daily average volume
        # Need to estimate 12h volume from daily volume (approximation)
        vol_filter = volume[i] > vol_sma_20_1d_aligned[i] * 0.5  # 12h is ~1/2 of 1d
        
        # Long conditions:
        # 1. Price breaks above R3 (bullish breakout)
        # 2. Price above 1w EMA20 (bullish trend)
        # 3. Volume confirmation
        if (close[i] > r3_aligned[i] and 
            close[i] > ema_20_1w_aligned[i] and 
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below S3 (bearish breakout)
        # 2. Price below 1w EMA20 (bearish trend)
        # 3. Volume confirmation
        elif (close[i] < s3_aligned[i] and 
              close[i] < ema_20_1w_aligned[i] and 
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_Trend_Vol"
timeframe = "12h"
leverage = 1.0