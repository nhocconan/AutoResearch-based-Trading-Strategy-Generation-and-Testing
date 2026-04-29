#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla R3/S3 provides strong intraday support/resistance levels
# 1d EMA34 provides higher timeframe trend filter (bullish when price > EMA34, bearish when price < EMA34)
# Volume spike (>2.0x 20-bar average) confirms breakout validity
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extreme levels

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    
    # Use previous day's OHLC for Camarilla calculation (no look-ahead)
    # We need to get the previous day's values for each 12h bar
    # Since we're on 12h timeframe, we look back 2 bars for previous day (approx)
    # But better: use HTF data to get proper daily levels
    
    # For 12h timeframe, we calculate Camarilla based on previous 1d candle
    # We'll shift the 1d data by 1 to get previous day's levels
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low_1d = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prev_close_1d = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    # Calculate Camarilla levels for previous day
    # R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # But we focus on R3 and S3 as primary levels
    prev_range_1d = prev_high_1d - prev_low_1d
    camarilla_r3_1d = prev_close_1d + 1.25 * prev_range_1d
    camarilla_s3_1d = prev_close_1d - 1.25 * prev_range_1d
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need 34 for EMA + 1 for previous day shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price below Camarilla S3 OR price below 1d EMA34 (trend change)
            if curr_close < curr_s3 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla R3 OR price above 1d EMA34 (trend change)
            if curr_close > curr_r3 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND above 1d EMA34 AND volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 AND below 1d EMA34 AND volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals