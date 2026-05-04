#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for trend direction (more responsive than 1d) and Camarilla pivot levels from 1h for entry/exit
# Volume confirmation requires 1.5x average volume to ensure strong participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag on 1h timeframe
# Works in both bull and bear markets by following the 4h trend direction and using Camarilla for structure

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from 1h data (self-referential for 1h timeframe)
    # We need to calculate Camarilla levels for each 1h bar using previous 1h bar's OHLC
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_pp = np.full(n, np.nan)  # pivot point
    
    for i in range(1, n):
        # Use previous bar's OHLC to calculate current bar's Camarilla levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        camarilla_pp[i] = pivot
        camarilla_r1[i] = pivot + 1.1 * (prev_high - prev_low) * 1.1 / 2.0
        camarilla_s1[i] = pivot - 1.1 * (prev_high - prev_low) * 1.1 / 2.0
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(vol_ema_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Camarilla breakout with 4h trend filter
        # Long: Price breaks above R1 + volume spike + price above 4h EMA50 (uptrend)
        # Short: Price breaks below S1 + volume spike + price below 4h EMA50 (downtrend)
        if position == 0:
            if (close[i] > camarilla_r1[i] and volume_spike and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            elif (close[i] < camarilla_s1[i] and volume_spike and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S1 OR price below 4h EMA50 (trend change)
            if close[i] < camarilla_s1[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above R1 OR price above 4h EMA50 (trend change)
            if close[i] > camarilla_r1[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals