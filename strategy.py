#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA200 trend filter and volume confirmation
# Uses 12h timeframe to reduce trade frequency (target: 12-37 trades/year)
# Camarilla R3/S3 levels provide structured breakout points
# 1w EMA200 ensures alignment with weekly trend (works in bull/bear markets)
# Volume confirmation > 1.8x average filters weak breakouts
# Discrete position sizing (0.25) and mean reversion exit at Camarilla H3/L3 levels
# Designed for BTC/ETH edge with controlled trade frequency

name = "12h_Camarilla_R3S3_1wEMA200_VolumeSpike_v1"
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
    
    # Precompute session filter (08-20 UTC) - avoids low-volume periods
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
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
    # H3/L3 for mean reversion exits (stronger levels)
    # H3 = C + (H - L) * 1.1 / 6
    h3 = close_1d + (high_1d - low_1d) * 1.1 / 6.0
    # L3 = C - (H - L) * 1.1 / 6
    l3 = close_1d - (high_1d - low_1d) * 1.1 / 6.0
    
    # Use previous day's values (shift by 1) to avoid look-ahead
    pp_shifted = np.roll(pp, 1)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    h3_shifted = np.roll(h3, 1)
    l3_shifted = np.roll(l3, 1)
    pp_shifted[0] = np.nan
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    h3_shifted[0] = np.nan
    l3_shifted[0] = np.nan
    
    # Align 1d indicators to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_shifted)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_shifted)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_shifted)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Volume and 1w EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
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
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_ema200_1w = ema_200_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below L3 (mean reversion to stronger support)
            if curr_close < curr_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above H3 (mean reversion to stronger resistance)
            if curr_close > curr_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirmed = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above R3, 1w EMA200 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema200_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 1w EMA200 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema200_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals