#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3, 12h EMA50 up-trend, volume > 2.0x average
# Short when price breaks below Camarilla S3, 12h EMA50 down-trend, volume > 2.0x average
# Exit when price crosses the Camarilla pivot point (mean reversion)
# Uses discrete position sizing (0.25) and stricter volume confirmation (2.0x) to target 30-50 trades/year
# Designed to work in both bull and bear markets by following the higher timeframe trend

name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike_v3"
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
    
    # Get 12h data for Camarilla pivot levels and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (using previous bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla calculations based on previous 12h bar
    # Pivot point = (H + L + C) / 3
    pp = (high_12h + low_12h + close_12h) / 3.0
    # R3 = C + (H - L) * 1.1 / 4
    r3 = close_12h + (high_12h - low_12h) * 1.1 / 4.0
    # S3 = C - (H - L) * 1.1 / 4
    s3 = close_12h - (high_12h - low_12h) * 1.1 / 4.0
    
    # Use previous bar's values (shift by 1) to avoid look-ahead
    pp_shifted = np.roll(pp, 1)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    pp_shifted[0] = np.nan
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_shifted)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_shifted)
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
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_pp = pp_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below pivot point (mean reversion to pivot)
            if curr_close < curr_pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above pivot point (mean reversion to pivot)
            if curr_close > curr_pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (stricter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above R3, 12h EMA50 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema50_12h and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 12h EMA50 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema50_12h and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals