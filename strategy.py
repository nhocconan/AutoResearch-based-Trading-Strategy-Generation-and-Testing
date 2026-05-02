#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivots provide intraday support/resistance levels derived from prior 4h range
# 4h EMA50 establishes higher-timeframe trend to avoid counter-trend trades
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity
# Works in bull markets (break above R1 + 4h EMA50 up) and bear markets (break below S1 + 4h EMA50 down)
# Uses discrete position sizing (0.20) to minimize fee churn and control drawdown
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h EMA20 for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    # 4h data for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Camarilla pivot calculation using prior 4h bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 1h (wait for completed 4h bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above R1 with volume confirmation and uptrend
            if close[i] > camarilla_r1_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: Break below S1 with volume confirmation and downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price re-enters Camarilla range (between S1 and R1) OR trend changes to downtrend
            if camarilla_s1_aligned[i] <= close[i] <= camarilla_r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price re-enters Camarilla range (between S1 and R1) OR trend changes to uptrend
            if camarilla_s1_aligned[i] <= close[i] <= camarilla_r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals