#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses 1d EMA50 for strong trend alignment (reduces whipsaw in ranging markets)
# Volume spike > 2.0x 20-period average to ensure institutional participation
# Mean reversion exit at Camarilla R4/S4 levels for tighter risk/reward
# Discrete position sizing (0.25) to minimize fee churn
# Designed for lower trade frequency (<100/year) while maintaining edge in both bull and bear regimes

name = "4h_Camarilla_R3S3_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations based on previous day
    # Pivot point = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R3 = C + (H - L) * 1.1 / 4
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    # S3 = C - (H - L) * 1.1 / 4
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    # R4 = C + (H - L) * 1.1 / 2
    r4 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Use previous day's values (shift by 1) to avoid look-ahead
    pp_shifted = np.roll(pp, 1)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r4_shifted = np.roll(r4, 1)
    s4_shifted = np.roll(s4, 1)
    pp_shifted[0] = np.nan
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    r4_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align 1d indicators to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_shifted)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_shifted)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 1d EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_pp = pp_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below S4 (mean reversion to strong support)
            if curr_close < curr_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above R4 (mean reversion to strong resistance)
            if curr_close > curr_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above R3, 1d EMA50 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema50_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 1d EMA50 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema50_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals