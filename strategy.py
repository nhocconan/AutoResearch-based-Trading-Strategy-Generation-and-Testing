#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and 1d EMA34 trend filter.
# Enter long when price breaks above Camarilla R3, 1d volume > 2.0x 20-bar average, and close > 1d EMA34 (uptrend).
# Enter short when price breaks below Camarilla S3 under same conditions but close < 1d EMA34.
# Exit when price crosses Camarilla pivot point (PP) or volume drops below average.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Camarilla levels provide precise intraday support/resistance; volume spike confirms institutional interest; EMA34 filters for trend alignment.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume and its 20-period average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Volume spike: >2.0x 20-bar average
    volume_spike = vol_1d > 2.0 * vol_ma_20
    
    # Align 1d indicators to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Camarilla levels from previous 1d bar (using typical price)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    pp = typical_price_1d  # Pivot point
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla levels
    r3 = pp + range_1d * 1.1 / 4
    s3 = pp - range_1d * 1.1 / 4
    r4 = pp + range_1d * 1.1 / 2
    s4 = pp - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure sufficient history for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        vol_spike = volume_spike_aligned[i] > 0.5  # Boolean as float
        
        # Trend filter: close > EMA34 for long, close < EMA34 for short
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r3_aligned[i]
        breakout_down = close[i] < s3_aligned[i]
        
        # Exit conditions: price crosses pivot point or volume drops
        exit_long = close[i] < pp_aligned[i]
        exit_short = close[i] > pp_aligned[i]
        
        # Handle entries and exits
        if breakout_up and uptrend and vol_spike and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and downtrend and vol_spike and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals