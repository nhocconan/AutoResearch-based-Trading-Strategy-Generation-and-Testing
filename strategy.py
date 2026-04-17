#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d Volume Spike + Weekly Trend Filter.
Long when Williams %R < -80 (oversold) AND 1d volume > 1.5x 20-period average AND price > 1w EMA50 (weekly uptrend).
Short when Williams %R > -20 (overbought) AND 1d volume > 1.5x 20-period average AND price < 1w EMA50 (weekly downtrend).
Exit when Williams %R reverses (> -50 for longs, < -50 for shorts) or weekly trend reverses.
Uses 1d for volume spike detection, 1w for EMA50 trend filter, 6h for Williams %R timing.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures mean reversion extremes,
volume spike confirms institutional participation, weekly EMA50 filters for higher-timeframe trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume spike detection
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike: current volume > 1.5x 20-period average
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20 = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R on 6h timeframe: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d volume spike and 1w EMA50 to 6h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(volume_spike_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # boolean check
        price = close[i]
        ema50 = ema50_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND volume spike AND price > 1w EMA50 (weekly uptrend)
            if wr < -80 and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND volume spike AND price < 1w EMA50 (weekly downtrend)
            elif wr > -20 and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -50 (reversing from oversold) OR price < 1w EMA50 (trend reversal)
            if wr > -50 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -50 (reversing from overbought) OR price > 1w EMA50 (trend reversal)
            if wr < -50 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_VolumeSpike_WeeklyEMA50_Trend"
timeframe = "6h"
leverage = 1.0