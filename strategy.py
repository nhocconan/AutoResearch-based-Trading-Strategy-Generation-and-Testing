#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 level and close > 1d EMA34 (uptrend) with volume > 2.0x average.
Short when price breaks below Camarilla S3 level and close < 1d EMA34 (downtrend) with volume > 2.0x average.
Exit on opposite Camarilla level break or trend reversal. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Camarilla R3/S3 levels provide stronger support/resistance than R1/S1, reducing false breakouts.
1d EMA34 filters medium-term trend, volume spike confirms breakout strength. Designed for both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Calculate Camarilla levels for today using previous day's OHLC
        # Need to get previous day's high, low, close from 1d data
        # We'll use the actual daily OHLC from the 1d dataframe
        # Find the index of the most recent completed 1d bar
        if i >= 6:  # Need at least 6*4h = 24h to get previous day
            # Get the timestamp of current bar
            current_time = prices['open_time'].iloc[i]
            # Find the 1d bar that ended before current_time
            # Since df_1d is already aligned, we can use the index
            # df_1d index corresponds to the start of each 1d bar
            # We need to find which 1d bar contains the current 4h bar
            # For simplicity, we'll use the last completed 1d bar
            # The 1d bar at index j ended at df_1d['open_time'].iloc[j] + 1 day
            # We want the 1d bar that ended most recently before current_time
            
            # Convert to datetime for comparison
            current_dt = pd.Timestamp(current_time)
            
            # Find the index of the last 1d bar that ended before current_dt
            # df_1d['open_time'] contains the start of each 1d bar
            # The bar ended at start + 1 day
            ended_times = df_1d['open_time'] + pd.Timedelta(days=1)
            # Get indices where ended_times < current_dt
            valid_indices = np.where(ended_times.values < current_dt)[0]
            
            if len(valid_indices) > 0:
                # Get the most recent completed 1d bar
                idx_1d = valid_indices[-1]
                prev_high = df_1d['high'].iloc[idx_1d]
                prev_low = df_1d['low'].iloc[idx_1d]
                prev_close = df_1d['close'].iloc[idx_1d]
                
                # Camarilla levels
                range_val = prev_high - prev_low
                camarilla_r3 = prev_close + (range_val * 1.1 / 4)
                camarilla_s3 = prev_close - (range_val * 1.1 / 4)
            else:
                # Not enough 1d data yet
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > camarilla_r3 and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < camarilla_s3 and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 OR trend reversal
                if (price < camarilla_s3 or price < ema34_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R3 OR trend reversal
                if (price > camarilla_r3 or price > ema34_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3_S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0