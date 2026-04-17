#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 of daily close for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe (use previous day's values to avoid look-ahead)
    bull_power_1d_prev = np.roll(bull_power_1d, 1)
    bear_power_1d_prev = np.roll(bear_power_1d, 1)
    bull_power_1d_prev[0] = np.nan
    bear_power_1d_prev[0] = np.nan
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d_prev)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d_prev)
    
    # Calculate 6-period RSI on 6h close for momentum confirmation
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    avg_loss = loss.ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need volume MA20 and RSI warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        # RSI filter: avoid extremes (30 < RSI < 70) to reduce whipsaw
        rsi_filter = (rsi[i] > 30) & (rsi[i] < 70)
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) with volume and RSI filter
            if bull_power_aligned[i] > 0 and volume_filter and rsi_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish momentum) with volume and RSI filter
            elif bear_power_aligned[i] < 0 and volume_filter and rsi_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative or volume/RSI filter fails
            if bull_power_aligned[i] <= 0 or not volume_filter or not rsi_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive or volume/RSI filter fails
            if bear_power_aligned[i] >= 0 or not volume_filter or not rsi_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_RSI_VolumeFilter"
timeframe = "6h"
leverage = 1.0