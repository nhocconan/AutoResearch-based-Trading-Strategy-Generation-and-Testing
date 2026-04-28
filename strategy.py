#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction and 4h Camarilla R3/S3 breakout with volume confirmation.
# Enter long when price is above 12h Supertrend (bullish trend) and breaks above 4h Camarilla R3 with volume spike.
# Enter short when price is below 12h Supertrend (bearish trend) and breaks below 4h Camarilla S3 with volume spike.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 20-50 trades/year.
# Supertrend provides robust trend filtering, Camarilla levels provide precise entry/exit points, volume confirms breakout strength.
# Works in bull (trend + breakouts) and bear (failed breaks reverse via trend filter) markets.

name = "4h_Camarilla_R3S3_Breakout_12hSupertrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR
    atr_period = 10
    atr = np.full_like(close_12h, np.nan)
    for i in range(atr_period, len(close_12h)):
        if i == atr_period:
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend
    multiplier = 3.0
    upperband = np.full_like(close_12h, np.nan)
    lowerband = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        upperband[i] = (high_12h[i] + low_12h[i]) / 2 + multiplier * atr[i]
        lowerband[i] = (high_12h[i] + low_12h[i]) / 2 - multiplier * atr[i]
    
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(supertrend[i-1]):
            # Initialize
            supertrend[i] = upperband[i]
            direction[i] = 1
        else:
            if close_12h[i] > supertrend[i-1]:
                supertrend[i] = upperband[i]
                direction[i] = 1
            else:
                supertrend[i] = lowerband[i]
                direction[i] = -1
    
    # Align 12h Supertrend direction to 4h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate 4h Camarilla pivots (using previous bar's high, low, close)
    n_4h = len(close)
    camarilla_r3 = np.full(n_4h, np.nan)
    camarilla_s3 = np.full(n_4h, np.nan)
    
    for i in range(1, n_4h):
        # Use previous bar to avoid look-ahead
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        pivot = (phigh + plow + pclose) / 3.0
        rng = phigh - plow
        camarilla_r3[i] = pivot + rng * 1.1 / 4.0
        camarilla_s3[i] = pivot - rng * 1.1 / 4.0
    
    # Forward fill Camarilla levels
    camarilla_r3 = pd.Series(camarilla_r3).ffill().values
    camarilla_s3 = pd.Series(camarilla_s3).ffill().values
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from Supertrend
        is_uptrend = supertrend_dir_aligned[i] == 1
        is_downtrend = supertrend_dir_aligned[i] == -1
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        long_breakout = is_uptrend and close[i] > camarilla_r3[i] and volume_spike[i]
        short_breakout = is_downtrend and close[i] < camarilla_s3[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_s3[i]
        short_exit = close[i] > camarilla_r3[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals