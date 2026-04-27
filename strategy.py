#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume spike
# Uses price channel breakouts from daily pivots, filtered by daily trend and volume.
# Works in bull/bear: Buys strength in uptrend, sells weakness in downtrend.
# Volume filter prevents false breakouts. Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if i < 2:  # Need at least 2 periods for Camarilla calculation
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels for previous day (using previous day's OHLC)
        idx_1d = i // 24  # 24 hours in a day at 1h intervals, but we use 4h so adjust
        # Better approach: use the actual 1d data index
        # Since we're on 4h timeframe, each day has 6 bars
        if len(df_1d) > 0:
            # Get the most recent completed day
            day_idx = min(len(df_1d) - 1, (i // 6))  # 6 bars per day on 4h
            if day_idx >= 1:
                prev_high = df_1d['high'].iloc[day_idx - 1]
                prev_low = df_1d['low'].iloc[day_idx - 1]
                prev_close = df_1d['close'].iloc[day_idx - 1]
                
                # Camarilla levels
                range_val = prev_high - prev_low
                r3 = prev_close + (range_val * 1.1 / 4)
                s3 = prev_close - (range_val * 1.1 / 4)
                
                # Long: price breaks above R3 with trend and volume
                if (close[i] > r3 and 
                    close[i] > ema50_1d_aligned[i] and 
                    volume_filter[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S3 with trend and volume
                elif (close[i] < s3 and 
                      close[i] < ema50_1d_aligned[i] and 
                      volume_filter[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0