#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot R4/S4 breakout with volume confirmation and ADX trend filter.
# Enter long when price breaks above 1w Camarilla R4 with volume spike and ADX > 25 (strong trend).
# Enter short when price breaks below 1w Camarilla S4 with volume spike and ADX > 25.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 12-37 trades/year.
# Weekly Camarilla provides major structure from higher timeframe, volume confirms breakout strength,
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# Works in bull (breakouts with trend) and bear (failed breaks reverse via exits) markets.

name = "6h_Camarilla_R4S4_Breakout_Volume_ADXFilter_v1"
timeframe = "6h"
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
    camarilla_r4 = np.full(n_1w, np.nan)
    camarilla_s4 = np.full(n_1w, np.nan)
    
    for i in range(1, n_1w):
        # Use previous bar to avoid look-ahead
        phigh = high_1w[i-1]
        plow = low_1w[i-1]
        pclose = close_1w[i-1]
        pivot = (phigh + plow + pclose) / 3.0
        rng = phigh - plow
        camarilla_r4[i] = pivot + rng * 1.1 / 2.0  # R4 level
        camarilla_s4[i] = pivot - rng * 1.1 / 2.0  # S4 level
    
    # Forward fill Camarilla levels
    camarilla_r4 = pd.Series(camarilla_r4).ffill().values
    camarilla_s4 = pd.Series(camarilla_s4).ffill().values
    
    # Align 1w indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Calculate 6h ADX (14) for trend strength
    def calculate_adx(high, low, close, length=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # First value has no previous close
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/length)
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # Initial values (simple average)
        if len(tr) >= length:
            atr[length-1] = np.mean(tr[1:length])
            dm_plus_smooth[length-1] = np.mean(dm_plus[1:length])
            dm_minus_smooth[length-1] = np.mean(dm_minus[1:length])
        
        # Wilder's smoothing for subsequent values
        for i in range(length, len(tr)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (length-1) + dm_plus[i]) / length
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (length-1) + dm_minus[i]) / length
        
        # Directional Indicators
        di_plus = np.zeros_like(close)
        di_minus = np.zeros_like(close)
        dx = np.zeros_like(close)
        
        # Avoid division by zero
        valid_atr = atr != 0
        di_plus[valid_atr] = (dm_plus_smooth[valid_atr] / atr[valid_atr]) * 100
        di_minus[valid_atr] = (dm_minus_smooth[valid_atr] / atr[valid_atr]) * 100
        
        # DX and ADX
        di_sum = di_plus + di_minus
        valid_di_sum = di_sum != 0
        dx[valid_di_sum] = (np.abs(di_plus[valid_di_sum] - di_minus[valid_di_sum]) / di_sum[valid_di_sum]) * 100
        
        # ADX (smoothed DX)
        adx = np.zeros_like(close)
        if len(dx) >= length:
            adx[length-1] = np.mean(dx[1:length])
            for i in range(length, len(dx)):
                adx[i] = (adx[i-1] * (length-1) + dx[i]) / length
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    strong_trend = adx > 25  # Strong trend when ADX > 25
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions with volume confirmation and ADX trend filter
        long_breakout = close[i] > camarilla_r4_aligned[i] and volume_spike[i] and strong_trend[i]
        short_breakout = close[i] < camarilla_s4_aligned[i] and volume_spike[i] and strong_trend[i]
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_s4_aligned[i]
        short_exit = close[i] > camarilla_r4_aligned[i]
        
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