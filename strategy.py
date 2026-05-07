#!/usr/bin/env python3
"""
1D_Weekly_TRIX_Volume_Spike_Regime_v1
Hypothesis: Use weekly TRIX for trend direction and daily TRIX for entry timing.
Long when weekly TRIX > 0, daily TRIX crosses above -0.05, and volume > 1.5x 20-day average.
Short when weekly TRIX < 0, daily TRIX crosses below 0.05, and volume > 1.5x 20-day average.
Exit when daily TRIX crosses back toward zero or volume dries up.
Weekly TRIX filters for primary trend, daily TRIX provides timely entries, volume confirms momentum.
Designed to work in both bull (follow weekly uptrend) and bear (follow weekly downtrend) markets.
"""
name = "1D_Weekly_TRIX_Volume_Spike_Regime_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly TRIX (15-period EMA of EMA of EMA of close)
    close_weekly = pd.Series(df_weekly['close'])
    ema1 = close_weekly.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix_weekly = (ema3.diff() / ema3.shift(1)) * 100
    trix_weekly = trix_weekly.fillna(0).values
    trix_weekly_aligned = align_htf_to_ltf(prices, df_weekly, trix_weekly)
    
    # Calculate daily TRIX for entry signals
    close_daily = pd.Series(close)
    ema1_d = close_daily.ewm(span=15, adjust=False).mean()
    ema2_d = ema1_d.ewm(span=15, adjust=False).mean()
    ema3_d = ema2_d.ewm(span=15, adjust=False).mean()
    trix_daily = (ema3_d.diff() / ema3_d.shift(1)) * 100
    trix_daily = trix_daily.fillna(0).values
    
    # Volume filter: current volume > 1.5 * 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(30, 20)  # Ensure sufficient warmup for TRIX
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(trix_weekly_aligned[i]) or np.isnan(trix_daily[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 3 days between trades to reduce frequency
            if bars_since_exit < 3:
                continue
                
            # Long: weekly trend up, daily TRIX crosses above -0.05, volume spike
            if (trix_weekly_aligned[i] > 0 and 
                trix_daily[i] > -0.05 and trix_daily[i-1] <= -0.05 and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: weekly trend down, daily TRIX crosses below 0.05, volume spike
            elif (trix_weekly_aligned[i] < 0 and 
                  trix_daily[i] < 0.05 and trix_daily[i-1] >= 0.05 and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: TRIX crosses back below 0 or volume dries up
                if trix_daily[i] < 0 or not volume_filter[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: TRIX crosses back above 0 or volume dries up
                if trix_daily[i] > 0 or not volume_filter[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals