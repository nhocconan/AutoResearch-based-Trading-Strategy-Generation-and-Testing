#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3, 1w EMA34 up-trend, volume > 2.0x average
# Short when price breaks below 1d Camarilla S3, 1w EMA34 down-trend, volume > 2.0x average
# Exit when price crosses the 50% level (midpoint of Camarilla width: (R4-S4)/2)
# Uses discrete position sizing (0.25) and strong volume filter to target 12-37 trades/year.
# Designed to work in both bull and bear markets by following the higher timeframe trend.

name = "12h_Camarilla_R3S3_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla pivot levels (R3, S3, R4, S4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla width and levels
    camarilla_width = (high_1d - low_1d) * 1.1 / 12.0  # Standard Camarilla multiplier
    camarilla_r3 = close_1d + camarilla_width * 3.0
    camarilla_s3 = close_1d - camarilla_width * 3.0
    camarilla_r4 = close_1d + camarilla_width * 4.0
    camarilla_s4 = close_1d - camarilla_width * 4.0
    camarilla_mid = (camarilla_r4 + camarilla_s4) / 2.0  # Midpoint for exit
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume and 1w EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_mid = camarilla_mid_aligned[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 50% level (mean reversion to midpoint)
            if curr_close < curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 50% level (mean reversion to midpoint)
            if curr_close > curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (strong filter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above R3, 1w EMA34 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema34_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 1w EMA34 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema34_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals