#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Long when: price breaks above Camarilla R3 AND 1d close > 1d EMA34 AND 6h volume > 1.8x 20-period average
# Short when: price breaks below Camarilla S3 AND 1d close < 1d EMA34 AND 6h volume > 1.8x 20-period average
# Uses Camarilla pivot levels for structure, 1d EMA34 for trend alignment, volume spike for conviction.
# Target: 12-37 trades/year on 6h. Discrete sizing 0.25 to balance return and fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 1d trend.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for price action and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Ranges
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2.0)  # R3
    s3 = pivot - (range_hl * 1.1 / 2.0)  # S3
    
    # Align Camarilla levels to 6h primary timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h volume average (20-period) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for 1d EMA34
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.8x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # 1d trend filter
        uptrend_1d = curr_close > curr_ema_34
        downtrend_1d = curr_close < curr_ema_34
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3 AND 1d uptrend AND volume confirmation
            if (curr_high > curr_r3 and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 AND 1d downtrend AND volume confirmation
            elif (curr_low < curr_s3 and 
                  downtrend_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla R3 (breakdown) OR 1d trend turns down
            if (curr_close < curr_r3 or 
                not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla S3 (breakout) OR 1d trend turns up
            if (curr_close > curr_s3 or 
                not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals