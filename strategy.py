#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation
# Long when price breaks above R3 AND close > 4h EMA50 (uptrend) AND 1d volume > 2.0 * 20-bar avg volume
# Short when price breaks below S3 AND close < 4h EMA50 (downtrend) AND 1d volume > 2.0 * 20-bar avg volume
# Exit when price retraces to the Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete sizing 0.20 to balance return and fee drag
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# 4h EMA50 provides intermediate trend filter between 1h and 1d for better regime adaptation
# 1d volume spike confirms institutional participation and reduces false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Pivot exit works in ranging markets and captures mean reversion after breakout failure

name = "1h_Camarilla_R3S3_4hEMA50_1dVolumeSpike_Session_v2"
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
    
    # Calculate Camarilla pivot levels for 1h timeframe (based on previous bar)
    # Camarilla: Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1 / 2
    # S3 = Pivot - (H - L) * 1.1 / 2
    pivot = (high + low + close) / 3.0
    r3 = pivot + (high - low) * 1.1 / 2.0
    s3 = pivot - (high - low) * 1.1 / 2.0
    
    # Shift by 1 to use only completed bar data (no look-ahead)
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    pivot_prev = np.roll(pivot, 1)
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan
    pivot_prev[0] = np.nan
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data ONCE before loop for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    
    # Align 1d volume spike to 1h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_prev[i]) or np.isnan(s3_prev[i]) or 
            np.isnan(pivot_prev[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: Break above R3 AND uptrend AND volume spike
            if close[i] > r3_prev[i] and close[i] > ema50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Break below S3 AND downtrend AND volume spike
            elif close[i] < s3_prev[i] and close[i] < ema50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price retraces to pivot point (mean reversion)
            if close[i] <= pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price retraces to pivot point (mean reversion)
            if close[i] >= pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals