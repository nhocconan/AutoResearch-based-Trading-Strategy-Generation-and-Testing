#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Long when price breaks above 1h Camarilla R3, 4h EMA34 up-trend, volume > 1.5x average
# Short when price breaks below 1h Camarilla S3, 4h EMA34 down-trend, volume > 1.5x average
# Exit when price crosses the Camarilla pivot point (mean reversion)
# Uses discrete position sizing (0.20) and session filter (08-20 UTC) to target 15-37 trades/year.
# Designed to work in both bull and bear markets by following the 4h trend.

name = "1h_Camarilla_R3S3_4hEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1h data for Camarilla levels (using previous day's OHLC)
    df_1h = get_htf_data(prices, '1d')  # Need daily OHLC for Camarilla calculation
    if len(df_1h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: based on previous day's high, low, close
    prev_high = df_1h['high'].values
    prev_low = df_1h['low'].values
    prev_close = df_1h['close'].values
    
    # Camarilla levels
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r3 = camarilla_pivot + (camarilla_range * 1.1 / 4.0)  # R3 = pivot + 1.1*range/4
    camarilla_s3 = camarilla_pivot - (camarilla_range * 1.1 / 4.0)  # S3 = pivot - 1.1*range/4
    camarilla_r4 = camarilla_pivot + (camarilla_range * 1.1 / 2.0)  # R4 = pivot + 1.1*range/2
    camarilla_s4 = camarilla_pivot - (camarilla_range * 1.1 / 2.0)  # S4 = pivot - 1.1*range/2
    
    # Align Camarilla levels to 1h timeframe (no additional delay needed for daily-based)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1h, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s4)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume and 4h EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        curr_ema34_4h = ema_34_4h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below pivot point (mean reversion to pivot)
            if curr_close < curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price above pivot point (mean reversion to pivot)
            if curr_close > curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above R3, 4h EMA34 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema34_4h and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3, 4h EMA34 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema34_4h and vol_confirmed:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals