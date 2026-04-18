#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with daily volume spike and trend filter.
# Camarilla levels (H3/L3) act as strong intraday support/resistance.
# Daily volume spike (>1.5x average) confirms institutional interest.
# 1-day EMA34 filter ensures trades align with higher timeframe trend.
# Designed for low trade frequency (20-40/year) to minimize fee drag.
# Works in bull markets (buy dips at L3 in uptrend) and bear markets (sell rallies at H3 in downtrend).

name = "4h_Camarilla_L3H3_VolumeSpike_EMA34Filter"
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
    
    # Get daily data for Camarilla calculation and EMA filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # H3 = C + (H-L)*1.1/6, L3 = C - (H-L)*1.1/6
    # Using previous day's data to avoid look-ahead
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            rang = high_1d[i-1] - low_1d[i-1]
            camarilla_h3[i] = close_1d[i-1] + rang * 1.1 / 6
            camarilla_l3[i] = close_1d[i-1] - rang * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 20-period average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5x 20-period average
        vol_spike = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: price at L3 with volume spike and above daily EMA34 (uptrend)
            long_condition = (close[i] <= camarilla_l3_aligned[i] * 1.001) and vol_spike and (close[i] > ema34_aligned[i])
            if long_condition:
                signals[i] = 0.25
                position = 1
            # Short: price at H3 with volume spike and below daily EMA34 (downtrend)
            elif (close[i] >= camarilla_h3_aligned[i] * 0.999) and vol_spike and (close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches H3 or closes below daily EMA34
            exit_condition = (close[i] >= camarilla_h3_aligned[i] * 0.999) or (close[i] < ema34_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches L3 or closes above daily EMA34
            exit_condition = (close[i] <= camarilla_l3_aligned[i] * 1.001) or (close[i] > ema34_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals