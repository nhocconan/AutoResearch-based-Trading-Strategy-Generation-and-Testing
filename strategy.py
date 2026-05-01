#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation.
# Long when: price breaks above Camarilla R3 (1h) AND 4h close > 4h EMA50 AND 1h volume > 1.5x 20-period average
# Short when: price breaks below Camarilla S3 (1h) AND 4h close < 4h EMA50 AND 1h volume > 1.5x 20-period average
# Uses Camarilla pivots for intraday structure, 4h EMA50 for trend alignment, volume spike for conviction.
# Target: 15-37 trades/year on 1h. Discrete sizing 0.20 to minimize fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 4h trend.
# Session filter (08-20 UTC) reduces noise and overtrading.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1h data ONCE before loop for Camarilla calculation and volume
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 1h Camarilla levels (based on previous day)
    # For simplicity, use rolling window of 1h bars approximating prior day (24h = 24 bars)
    lookback = 24  # approximately 1 day of 1h bars
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    rng = high_1h - low_1h
    camarilla_r3 = close_1h + 1.1 * rng * 1.1 / 4.0
    camarilla_s3 = close_1h - 1.1 * rng * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe (already aligned as calculated on 1h)
    # But we need to shift by 1 bar to avoid look-ahead (use previous bar's levels)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r3_shifted[0] = np.nan  # first bar has no prior
    camarilla_s3_shifted[0] = np.nan
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h volume average (20-period) for volume confirmation
    vol_1h = df_1h['volume'].values
    vol_ma_1h = pd.Series(vol_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # warmup for 4h EMA50 and 1h lookback
    
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
        if (np.isnan(camarilla_r3_shifted[i]) or np.isnan(camarilla_s3_shifted[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_1h_aligned[i]
        curr_r3 = camarilla_r3_shifted[i]
        curr_s3 = camarilla_s3_shifted[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 4h trend filter
        uptrend_4h = curr_close > curr_ema_50
        downtrend_4h = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3 AND 4h uptrend AND volume confirmation
            if (curr_high > curr_r3 and 
                uptrend_4h and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: break below Camarilla S3 AND 4h downtrend AND volume confirmation
            elif (curr_low < curr_s3 and 
                  downtrend_4h and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla R3 (breakdown) OR 4h trend turns down
            if (curr_close < curr_r3 or 
                not uptrend_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla S3 (breakout) OR 4h trend turns up
            if (curr_close > curr_s3 or 
                not downtrend_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals