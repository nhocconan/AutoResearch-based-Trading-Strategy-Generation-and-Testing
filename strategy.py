#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla R3/S3 breakout with weekly trend filter and volume spike
# Long when price breaks above R3, weekly EMA(34) uptrend, and volume spike
# Short when price breaks below S3, weekly EMA(34) downtrend, and volume spike
# Uses weekly trend filter to capture major trends while avoiding counter-trend trades
# Targets 75-200 total trades over 4 years (19-50/year) for optimal balance of opportunity and cost

name = "4h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Standard Camarilla formula: R3 = C + (H-L)*1.1/6, S3 = C - (H-L)*1.1/6
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero and handle first bar
    price_range = prev_high - prev_low
    # Camarilla multipliers
    r3 = prev_close + price_range * 1.1 / 6  # R3 = C + (H-L)*1.1/6
    s3 = prev_close - price_range * 1.1 / 6  # S3 = C - (H-L)*1.1/6
    
    # Align Camarilla levels to 4h timeframe (available after daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3, weekly uptrend, volume spike
            if price > r3_val and price > ema34_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, weekly downtrend, volume spike
            elif price < s3_val and price < ema34_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below R3 or weekly trend turns down
            if price < r3_val or price < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above S3 or weekly trend turns up
            if price > s3_val or price > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals