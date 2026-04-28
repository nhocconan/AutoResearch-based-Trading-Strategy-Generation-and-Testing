#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction, 1d ATR-based volatility breakout for entry timing, and volume confirmation.
# Enter long when price breaks above 1d ATR-based upper band with volume spike and 12h Supertrend uptrend.
# Enter short when price breaks below 1d ATR-based lower band with volume spike and 12h Supertrend downtrend.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 20-50 trades/year.
# Supertrend provides HTF trend filter, ATR bands adapt to volatility, volume confirms breakout strength.
# Works in bull (trend-following breaks) and bear (failed breaks reverse via trend filter) markets.

name = "4h_Supertrend12h_ATRBreakout1d_Volume_v1"
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
    
    # Get 12h data for Supertrend (HTF trend filter)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (10, 3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    n_12h = len(high_12h)
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range and ATR
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr = np.full(n_12h, np.nan)
    for i in range(atr_period, n_12h):
        if i == atr_period:
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate Supertrend
    hl2 = (high_12h + low_12h) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    supertrend = np.full(n_12h, np.nan)
    direction = np.full(n_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, n_12h):
        if i == atr_period:
            supertrend[i] = upperband[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == upperband[i-1]:
                if close_12h[i] <= upperband[i]:
                    supertrend[i] = upperband[i]
                else:
                    supertrend[i] = lowerband[i]
                    direction[i] = -1
            else:
                if close_12h[i] >= lowerband[i]:
                    supertrend[i] = lowerband[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upperband[i]
                    direction[i] = -1
    
    # Align 12h Supertrend direction to 4h timeframe
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Get 1d data for ATR-based volatility breakout
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR-based volatility breakout bands
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    atr_period_1d = 14
    multiplier_1d = 2.0
    
    # Calculate True Range and ATR for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_1d = np.full(n_1d, np.nan)
    for i in range(atr_period_1d, n_1d):
        if i == atr_period_1d:
            atr_1d[i] = np.nanmean(tr_1d[i-atr_period_1d+1:i+1])
        else:
            atr_1d[i] = (atr_1d[i-1] * (atr_period_1d - 1) + tr_1d[i]) / atr_period_1d
    
    # Calculate ATR-based breakout bands (using previous bar to avoid look-ahead)
    upper_band = np.full(n_1d, np.nan)
    lower_band = np.full(n_1d, np.nan)
    
    for i in range(1, n_1d):
        atr_val = atr_1d[i-1]
        if np.isnan(atr_val):
            continue
        upper_band[i] = close_1d[i-1] + multiplier_1d * atr_val
        lower_band[i] = close_1d[i-1] - multiplier_1d * atr_val
    
    # Forward fill bands
    upper_band = pd.Series(upper_band).ffill().values
    lower_band = pd.Series(lower_band).ffill().values
    
    # Align 1d indicators to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Calculate 4h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(direction_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > upper_band_aligned[i] and volume_spike[i] and direction_aligned[i] == 1
        short_breakout = close[i] < lower_band_aligned[i] and volume_spike[i] and direction_aligned[i] == -1
        
        # Exit conditions: opposite band or trend change
        long_exit = close[i] < lower_band_aligned[i] or direction_aligned[i] == -1
        short_exit = close[i] > upper_band_aligned[i] or direction_aligned[i] == 1
        
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