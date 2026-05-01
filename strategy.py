#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend and volume spike filter.
# Long when: price breaks above Camarilla R3 AND 12h close > 12h EMA50 AND 6h volume > 2.5x 20-period average
# Short when: price breaks below Camarilla S3 AND 12h close < 12h EMA50 AND 6h volume > 2.5x 20-period average
# Uses Camarilla levels from 1d data for structure, 12h EMA50 for trend alignment, volume spike for conviction.
# Target: 12-37 trades/year on 6h (50-150 total over 4 years). Discrete sizing 0.25 to minimize fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 12h trend.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 6h primary timeframe (no additional delay needed)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for 12h EMA50 (need 50+ for safety)
    
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
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_50 = ema_50_12h_aligned[i]
        
        # Volume confirmation: current 6h volume > 2.5x 20-period average
        # Calculate 6h volume MA on the fly
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        curr_vol_ma = vol_ma[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.5)
        
        # 12h trend filter
        uptrend_12h = curr_close > curr_ema_50
        downtrend_12h = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3 AND 12h uptrend AND volume confirmation
            if (curr_high > curr_r3 and 
                uptrend_12h and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 AND 12h downtrend AND volume confirmation
            elif (curr_low < curr_s3 and 
                  downtrend_12h and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla S3 (reversal) OR 12h trend turns down
            if (curr_close < curr_s3 or 
                not uptrend_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R3 (reversal) OR 12h trend turns up
            if (curr_close > curr_r3 or 
                not downtrend_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals