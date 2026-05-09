#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Camarilla levels provide institutional support/resistance; breakout above R3 or below S3 with volume
# confirms institutional participation. 1d EMA34 filters for trend direction, avoiding counter-trend trades.
# Volume spike (>2x average) ensures momentum. Designed for low-frequency, high-conviction trades.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using rolling window of previous day's data (shifted by 1 to avoid look-ahead)
    # We need daily OHLC, so we resample conceptually but use actual 1d data from get_htf_data
    # Camarilla formulas: 
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low)
    # We'll use the previous completed day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using that day's OHLC for next day's levels)
    # But we must use only completed days, so we shift by 1
    # R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low)
    camarilla_r3 = close_1d_vals + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d_vals - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels for current day's trading)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need 34 periods for EMA13 and 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Price breaks above R3 AND uptrend (price > 1d EMA34) AND volume spike
            if close[i] > r3_level and close[i] > ema_1d and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S3 AND downtrend (price < 1d EMA34) AND volume spike
            elif close[i] < s3_level and close[i] < ema_1d and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price re-enters below R3 OR trend reverses (price < 1d EMA34)
            if close[i] < r3_level or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price re-enters above S3 OR trend reverses (price > 1d EMA34)
            if close[i] > s3_level or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals