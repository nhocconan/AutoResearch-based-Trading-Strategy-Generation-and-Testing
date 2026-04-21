#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Fade_VolumeFilter_V1
Hypothesis: Fade extreme Camarilla R3/S3 levels on 4h with volume confirmation (>1.5x 20-bar MA) works in both bull and bear markets for BTC/ETH. Uses 12h HTF for trend filter (price > EMA34 = bull bias, price < EMA34 = bear bias) to align with dominant trend. Target: 30-60 trades/year per symbol (120-240 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data once for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Camarilla levels on 4h (primary timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Previous bar's OHLC for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    # Camarilla calculation: range = prev_high - prev_low
    rng = prev_high - prev_low
    # R3 = prev_close + rng * 1.1/4, S3 = prev_close - rng * 1.1/4
    r3 = prev_close + rng * 1.1 / 4
    s3 = prev_close - rng * 1.1 / 4
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        prev_close_price = prev_close[i]
        
        # Volume confirmation (>1.5x average)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Trend filter from 12h EMA34
        bull_trend = price > ema34_12h_aligned[i]
        bear_trend = price < ema34_12h_aligned[i]
        
        if position == 0:
            # Long fade: price crosses below S3 (support) in bear trend with volume
            if price < s3[i] and prev_close_price >= s3[i-1]:
                if bear_trend and volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Short fade: price crosses above R3 (resistance) in bull trend with volume
            elif price > r3[i] and prev_close_price <= r3[i-1]:
                if bull_trend and volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price crosses above S3 or reverse signal
            if price > s3[i] and prev_close_price <= s3[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below R3 or reverse signal
            if price < r3[i] and prev_close_price >= r3[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Fade_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0