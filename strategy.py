#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3S3 breakout with 4h trend filter and volume confirmation
# Long when price breaks above R3, above 4h EMA50, and volume spike
# Short when price breaks below S3, below 4h EMA50, and volume spike
# Exit when price returns to Pivot point or closes opposite side of S1/R1
# Uses Camarilla for intraday support/resistance, 4h EMA for trend filter, volume for confirmation
# Designed to work in trending markets via EMA filter and in ranging markets via mean reversion to pivot
# Target: 60-150 total trades over 4 years (15-37/year) with size 0.20

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_spike[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        # Need to get previous day's data - we'll use rolling window of 24h (24 candles for 1h)
        if i < 24:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get previous day's OHLC (24 hours ago)
        prev_high = np.max(high[i-24:i])
        prev_low = np.min(low[i-24:i])
        prev_close = close[i-24]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        pivot = (prev_high + prev_low + prev_close) / 3
        r3 = pivot + (prev_high - prev_low) * 1.1 / 2
        s3 = pivot - (prev_high - prev_low) * 1.1 / 2
        r1 = pivot + (prev_high - prev_low) * 1.1 / 6
        s1 = pivot - (prev_high - prev_low) * 1.1 / 6
        
        if position == 0:
            # Enter long: price breaks above R3, above 4h EMA50, volume spike
            if (close[i] > r3 and 
                close[i] > ema50_4h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S3, below 4h EMA50, volume spike
            elif (close[i] < s3 and 
                  close[i] < ema50_4h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot or closes below S1
            if (close[i] <= pivot) or (close[i] < s1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to pivot or closes above R1
            if (close[i] >= pivot) or (close[i] > r1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals