#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Point breakout with 1d trend filter and volume confirmation
# Uses actual Camarilla levels calculated from previous day's high/low/close.
# Long when price breaks above R3 with 1d uptrend and volume spike.
# Short when price breaks below S3 with 1d downtrend and volume spike.
# Target: 20-30 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + ((high - low) * 1.5000)
    # R3 = close + ((high - low) * 1.2500)
    # R2 = close + ((high - low) * 1.1666)
    # R1 = close + ((high - low) * 1.0833)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.0833)
    # S2 = close - ((high - low) * 1.1666)
    # S3 = close - ((high - low) * 1.2500)
    # S4 = close - ((high - low) * 1.5000)
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.2500
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.2500
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Price breaks above R3 + 1d uptrend + volume spike
        if (close[i] > camarilla_r3_aligned[i] and 
            close[i] > ema50_1d_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Price breaks below S3 + 1d downtrend + volume spike
        elif (close[i] < camarilla_s3_aligned[i] and 
              close[i] < ema50_1d_aligned[i] and   # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0