#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_VolumeSpike
Hypothesis: 6h Donchian(20) breakouts in the direction of the 12h trend with volume confirmation capture sustained moves. 
In bull markets, upside breakouts with 12h uptrend go long. In bear markets, downside breakouts with 12h downtrend go short. 
Volume filter (>2.0x 20-bar average) prevents false breakouts in low volatility periods. 
Exits on opposite Donchian band touch or trend reversal. Discrete sizing (0.25) limits fee churn.
Target: 50-150 total trades over 4 years (12-37/year). Works across BTC/ETH/SOL.
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
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels and 12h data for trend filter
    df_6h = get_htf_data(prices, '6h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 6h Donchian(20): upper/lower bands
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian upper: max(high, 20)
    upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: min(low, 20)
    lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (6h)
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper_6h)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower_6h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (tighter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Donchian (20), EMA50 (50), volume avg (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_12h_val = ema_50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine 12h trend: price > EMA50 = uptrend, price < EMA50 = downtrend
            is_uptrend = close_val > ema_12h_val
            is_downtrend = close_val < ema_12h_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above upper band and volume confirms
                if (close_val > upper_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below lower band and volume confirms
                if (close_val < lower_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches lower band (support) or trend changes to downtrend
            exit_condition = (close_val < lower_val) or (close_val < ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches upper band (resistance) or trend changes to uptrend
            exit_condition = (close_val > upper_val) or (close_val > ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0