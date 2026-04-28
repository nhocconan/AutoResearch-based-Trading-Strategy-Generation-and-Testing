#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w Camarilla pivot R3/S3 breakout with volume confirmation and ADX trend filter.
# Enter long when price breaks above 1w Camarilla R3 with volume spike and ADX > 25 (strong trend).
# Enter short when price breaks below 1w Camarilla S3 with volume spike and ADX > 25.
# Uses discrete position sizing (0.25) to reduce fee churn. Target: 20-40 trades/year.
# Weekly Camarilla provides major structure from higher timeframe, volume confirms breakout strength, ADX filter ensures trending conditions.
# Works in bull (breakouts with trend) and bear (failed breaks reverse via exits) markets by capturing strong directional moves.

name = "4h_Camarilla_R3S3_1w_Breakout_Volume_ADX_v1"
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
    
    # Get 1w data for Camarilla pivots (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivots (using previous bar's high, low, close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    camarilla_r3 = np.full(n_1w, np.nan)
    camarilla_s3 = np.full(n_1w, np.nan)
    
    for i in range(1, n_1w):
        # Use previous bar to avoid look-ahead
        phigh = high_1w[i-1]
        plow = low_1w[i-1]
        pclose = close_1w[i-1]
        pivot = (phigh + plow + pclose) / 3.0
        rng = phigh - plow
        camarilla_r3[i] = pivot + rng * 1.1 / 4.0
        camarilla_s3[i] = pivot - rng * 1.1 / 4.0
    
    # Forward fill Camarilla levels
    camarilla_r3 = pd.Series(camarilla_r3).ffill().values
    camarilla_s3 = pd.Series(camarilla_s3).ffill().values
    
    # Align 1w indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate 4h ADX (14) for trend strength
    def calculate_adx(high, low, close, length=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[length-1] = np.mean(tr[:length])
        plus_dm_smooth[length-1] = np.mean(plus_dm[:length])
        minus_dm_smooth[length-1] = np.mean(minus_dm[:length])
        
        for i in range(length, len(high)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (length-1) + plus_dm[i]) / length
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (length-1) + minus_dm[i]) / length
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        adx = np.zeros_like(dx)
        adx[2*length-2] = np.mean(dx[length-1:2*length-1])
        for i in range(2*length-1, len(high)):
            adx[i] = (adx[i-1] * (length-1) + dx[i]) / length
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    strong_trend = adx > 25  # Strong trend when ADX > 25
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions with volume confirmation and ADX trend filter
        long_breakout = close[i] > camarilla_r3_aligned[i] and volume_spike[i] and strong_trend[i]
        short_breakout = close[i] < camarilla_s3_aligned[i] and volume_spike[i] and strong_trend[i]
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_s3_aligned[i]
        short_exit = close[i] > camarilla_r3_aligned[i]
        
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