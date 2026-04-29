#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Uses 4h for signal direction (EMA50 trend) and 1h for entry timing precision
# Camarilla R3/S3 levels provide intraday support/resistance with lower false breakout rate than R4/S4
# Volume confirmation filters weak breakouts, reducing trade frequency
# Session filter (08-20 UTC) avoids low-liquidity periods
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session"
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
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate prior 4h OHLC for Camarilla levels (yesterday's 4h bar values)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Shift to get prior 4h bar's OHLC (avoid look-ahead)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    # First value is NaN (no prior 4h bar)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    # Calculate Camarilla R3/S3 levels based on prior 4h bar OHLC
    # Camarilla R3/S3: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = prev_high_4h - prev_low_4h
    camarilla_r3 = prev_close_4h + camarilla_range * 1.1 / 4
    camarilla_s3 = prev_close_4h - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 20-period average volume for confirmation (on 1h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session hours (08-20 UTC) for filtering
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 4h EMA50, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below Camarilla S3 OR price closes below 4h EMA50
            if curr_close < curr_s3 or curr_close < curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R3 OR price closes above 4h EMA50
            if curr_close > curr_r3 or curr_close > curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + price above 4h EMA50 + volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_4h and 
                vol_confirm):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla S3 + price below 4h EMA50 + volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_4h and 
                  vol_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals