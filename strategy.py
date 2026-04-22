#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly EMA50 trend filter with daily price action and volume confirmation
# Uses 1w EMA50 for trend direction, 1d price closes for entry/exit, and volume filter
# Designed to work in both bull and bear markets by following the weekly trend
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA50 - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Load daily data for price and volume averages
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    volume_daily = df_daily['volume'].values
    vol_avg_20 = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume average to 1d timeframe (no additional delay needed)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
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
            # Long: Price above weekly EMA50 with volume confirmation
            if (close[i] > ema50_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA50 with volume confirmation
            elif (close[i] < ema50_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back below/above weekly EMA50
            if position == 1:
                if close[i] < ema50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_WeeklyEMA50_Volume_Session"
timeframe = "1d"
leverage = 1.0