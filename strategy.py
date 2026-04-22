#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
Long when Bull Power > 0 and Bear Power < 0 (bullish divergence) with price above 1d EMA50 and volume spike.
Short when Bear Power > 0 and Bull Power < 0 (bearish divergence) with price below 1d EMA50 and volume spike.
Exit when Bull Power and Bear Power converge (cross zero) or opposing signal appears.
Designed for low trade frequency (15-30/year) to minimize fee drift in ranging markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter and Elder Ray calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Elder Ray components (13-period EMA for consistency)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d_series = pd.Series(df_1d['close'].values)
    
    ema13 = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d.values - ema13  # Bull Power = High - EMA13
    bear_power = low_1d.values - ema13   # Bear Power = Low - EMA13
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA lookback
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Bull Power > 0 and Bear Power < 0 (bullish divergence) with uptrend and volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                close[i] > ema50_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 and Bull Power < 0 (bearish divergence) with downtrend and volume spike
            elif (bear_power_aligned[i] > 0 and 
                  bull_power_aligned[i] < 0 and 
                  close[i] < ema50_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: convergence of Bull/Bear Power or opposing signal
            exit_signal = False
            
            if position == 1:
                # Exit long: Bear Power becomes positive (loss of bullish momentum)
                if bear_power_aligned[i] >= 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bull Power becomes positive (loss of bearish momentum)
                if bull_power_aligned[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0
#%%