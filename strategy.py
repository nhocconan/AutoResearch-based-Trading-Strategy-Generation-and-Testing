#!/usr/bin/env python3

"""
Hypothesis: 4-hour Volume-Weighted Average Price (VWAP) bands with 1-day trend filter and volume confirmation.
VWAP bands provide dynamic support/resistance based on volume-weighted price action.
The 1-day trend filter ensures trades align with the daily trend to avoid counter-trend trades.
Volume spikes confirm institutional participation at VWAP band touches.
This strategy aims to capture mean-reversion bounces from VWAP bands in both bull and bear markets
by trading reversals at VWAP bands with trend and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap_bands(high, low, close, volume, period=20):
    """Calculate VWAP and standard deviation bands"""
    typical_price = (high + low + close) / 3
    vwap = np.nancumsum(typical_price * volume) / np.nancumsum(volume)
    
    # Reset VWAP calculation periodically (every period)
    vwap_series = pd.Series(vwap)
    vwap_reset = vwap_series.rolling(window=period, min_periods=period).apply(
        lambda x: np.nansum(x * volume[len(x)-len(x):]) / np.nansum(volume[len(x)-len(x):]), raw=False
    )
    
    # Simpler approach: calculate VWAP over rolling window
    vwap_window = pd.Series(typical_price * volume).rolling(window=period, min_periods=period).sum() / \
                  pd.Series(volume).rolling(window=period, min_periods=period).sum()
    
    # Calculate standard deviation of price from VWAP
    variance = pd.Series(((typical_price - vwap_window) ** 2) * volume).rolling(window=period, min_periods=period).sum() / \
               pd.Series(volume).rolling(window=period, min_periods=period).sum()
    std_dev = np.sqrt(variance)
    
    upper_band = vwap_window + (std_dev * 1.5)
    lower_band = vwap_window - (std_dev * 1.5)
    
    return vwap_window.values, upper_band.values, lower_band.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h VWAP bands data - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate VWAP bands on 4h data
    vwap_4h, upper_4h, lower_4h = calculate_vwap_bands(
        df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, df_4h['volume'].values
    )
    
    # Align VWAP bands to 4h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter (21-period)
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(vwap_4h_aligned[i]) or np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or np.isnan(ema_21_1d_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower VWAP band, above 1d EMA, volume spike
            if (close[i] <= lower_4h_aligned[i] and                  # Price at or below lower band
                close[i] > ema_21_1d_aligned[i] and                 # Above 1d EMA (bullish trend)
                volume[i] > 1.8 * vol_avg_20[i]):                   # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price touches upper VWAP band, below 1d EMA, volume spike
            elif (close[i] >= upper_4h_aligned[i] and               # Price at or above upper band
                  close[i] < ema_21_1d_aligned[i] and               # Below 1d EMA (bearish trend)
                  volume[i] > 1.8 * vol_avg_20[i]):                 # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to VWAP or crosses 1d EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above VWAP or below 1d EMA
                if close[i] >= vwap_4h_aligned[i] or close[i] < ema_21_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below VWAP or above 1d EMA
                if close[i] <= vwap_4h_aligned[i] or close[i] > ema_21_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VWAP_Bands_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0