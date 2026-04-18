#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h EMA trend filter and daily volume confirmation
# Williams %R(14) identifies overbought/oversold conditions for mean reversion entries
# 4h EMA(34) filters for trend direction - only trade in direction of higher timeframe trend
# Daily volume > 1.5x 20-period average confirms institutional participation
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend)
# Target: 15-37 trades/year (60-150 total over 4 years) to avoid excessive fee drag
name = "1h_WilliamsR_EMA34_VolumeFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R(14) calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 4h EMA(34) for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Daily volume filter
    df_1d = get_htf_data(prices, '1d')
    vol_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(vol_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        williams_val = williams_r[i]
        ema_34_val = ema_34_4h_aligned[i]
        vol_20_val = vol_20_1d_aligned[i]
        volume_filter = volume[i] > (1.5 * vol_20_val)
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above 4h EMA (uptrend) + volume confirmation
            if williams_val < -80 and close[i] > ema_34_val and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short: Williams %R overbought (> -20) + price below 4h EMA (downtrend) + volume confirmation
            elif williams_val > -20 and close[i] < ema_34_val and volume_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or trend breaks (price below EMA)
            if williams_val > -50 or close[i] < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or trend breaks (price above EMA)
            if williams_val < -50 or close[i] > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals