#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h EMA50 trend filter and volume spike confirmation.
# Williams %R(14) measures overbought/oversold conditions.
# 4h EMA50 filter ensures we trade only in the direction of the 4h trend.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend).
# Session filter (08-20 UTC) reduces noise trades.
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drift.
name = "1h_WilliamsR_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA50 filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema50_4h_aligned[i]
        downtrend = close[i] < ema50_4h_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend AND volume spike
            if williams_r[i] < -80 and uptrend and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend AND volume spike
            elif williams_r[i] > -20 and downtrend and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 OR trend reverses
            if williams_r[i] > -50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 OR trend reverses
            if williams_r[i] < -50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals