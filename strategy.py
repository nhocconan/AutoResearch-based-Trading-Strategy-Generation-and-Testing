#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Long when: price breaks above Camarilla R3 AND 1w close > 1w EMA34 AND 4h volume > 1.5x 20-period average
# Short when: price breaks below Camarilla S3 AND 1w close < 1w EMA34 AND 4h volume > 1.5x 20-period average
# Uses Camarilla pivot levels for structure, 1w EMA34 for trend alignment, volume spike for conviction.
# Target: 25-40 trades/year on 4h. Discrete sizing 0.25 to balance return and fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 1w trend.

name = "4h_Camarilla_R3S3_1wEMA34_VolumeConfirm_v1"
timeframe = "4h"
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
    
    # Load 4h data ONCE before loop for price action and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w typical price (PP) for Camarilla levels
    typical_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3 = PP + range * 1.1/4, S3 = PP - range * 1.1/4
    camarilla_pp = typical_1w
    camarilla_r3 = camarilla_pp + (range_1w * 1.1 / 4.0)
    camarilla_s3 = camarilla_pp - (range_1w * 1.1 / 4.0)
    
    # Align Camarilla levels to 4h primary timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 4h volume average (20-period) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for 1w EMA34
    
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
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 1w trend filter
        uptrend_1w = curr_close > curr_ema_34
        downtrend_1w = curr_close < curr_ema_34
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3 AND 1w uptrend AND volume confirmation
            if (curr_high > curr_r3 and 
                uptrend_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 AND 1w downtrend AND volume confirmation
            elif (curr_low < curr_s3 and 
                  downtrend_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla R3 (breakdown) OR 1w trend turns down
            if (curr_close < curr_r3 or 
                not uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla S3 (breakout) OR 1w trend turns up
            if (curr_close > curr_s3 or 
                not downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals