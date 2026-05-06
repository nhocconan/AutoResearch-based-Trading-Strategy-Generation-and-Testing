#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND 1d close > 1d EMA50 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below S3 AND 1d close < 1d EMA50 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price retraces to the Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete sizing 0.30 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d EMA50 provides strong trend filter between 12h and 1d for better regime adaptation
# Volume threshold increased to 2.0x to reduce false breakouts and lower trade frequency
# Camarilla pivot exit works in ranging markets and captures mean reversion after breakout failure
# This strategy focuses on BTC and ETH as primary targets, avoiding SOL-only bias

name = "12h_Camarilla_R3S3_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for 12h timeframe (based on previous bar)
    # Pivot = (high + low + close) / 3
    # R3 = pivot + (high - low) * 1.1 / 2
    # S3 = pivot - (high - low) * 1.1 / 2
    # R4 = pivot + (high - low) * 1.1
    # S4 = pivot - (high - low) * 1.1
    # We use R3/S3 for entries and pivot for exits
    typical_price = (high + low + close) / 3.0
    range_hl = high - low
    pivot = typical_price
    R3 = pivot + range_hl * 1.1 / 2.0
    S3 = pivot - range_hl * 1.1 / 2.0
    
    # Shift by 1 to use only completed bar data (no look-ahead)
    R3_prev = np.roll(R3, 1)
    S3_prev = np.roll(S3, 1)
    pivot_prev = np.roll(pivot, 1)
    R3_prev[0] = np.nan
    S3_prev[0] = np.nan
    pivot_prev[0] = np.nan
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(R3_prev[i]) or np.isnan(S3_prev[i]) or 
            np.isnan(pivot_prev[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: Break above R3 AND uptrend AND volume spike
            if close[i] > R3_prev[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Break below S3 AND downtrend AND volume spike
            elif close[i] < S3_prev[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price retraces to pivot point (mean reversion)
            if close[i] <= pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price retraces to pivot point (mean reversion)
            if close[i] >= pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals