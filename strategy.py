#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above R3, 12h EMA50 up-trend, volume > 2.0x average
# Short when price breaks below S3, 12h EMA50 down-trend, volume > 2.0x average
# Exit when price reverts to Camarilla midpoint (mean reversion)
# Uses discrete position sizing (0.25) and tight volume filter to limit trades to ~75-150 over 4 years.
# Uses 12h for signal direction/trend, 4h only for entry timing and breakout levels.
# This strategy targets 20-50 trades/year per symbol to avoid fee drag while capturing strong breakouts.

name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels (based on previous day)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels using previous day's OHLC
    # Shift by 1 to use completed day only (no look-ahead)
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h5 = prev_close + prev_range * 1.1/2  # R3
    camarilla_l5 = prev_close - prev_range * 1.1/2  # S3
    camarilla_mid = prev_close  # Pivot point
    
    # Align 4h indicators to 4h timeframe (no additional delay needed for Camarilla)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l5)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 12h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_h5 = camarilla_h5_aligned[i]
        curr_l5 = camarilla_l5_aligned[i]
        curr_mid = camarilla_mid_aligned[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Camarilla midpoint (mean reversion)
            if curr_close < curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla midpoint (mean reversion)
            if curr_close > curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (tight filter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above R3 (H5), 12h EMA50 up-trend, volume confirmed
            if curr_high > curr_h5 and curr_close > curr_ema50_12h and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 (L5), 12h EMA50 down-trend, volume confirmed
            elif curr_low < curr_l5 and curr_close < curr_ema50_12h and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals