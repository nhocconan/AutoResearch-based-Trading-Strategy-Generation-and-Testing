#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
# Uses 4h EMA50 for trend direction (bull/bear alignment) and 1d volume > 1.5x 20-period median for momentum confirmation.
# Enters on 1h breakouts above R3 (long) or below S3 (short) only when aligned with 4h trend and volume spike.
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years.
# Session filter: 08-20 UTC to avoid low-liquidity periods.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - avoid .dt accessor on datetime64
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume confirmation and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume median for confirmation (20-period)
    vol_median_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # Calculate Camarilla levels from previous day OHLC (1d)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels (breakout continuation zones)
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.05
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.05
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA50 and volume median
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_median_20_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 4h EMA50 direction
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Volume confirmation: current 1h volume > 1.5x 1d volume median (scaled)
        # Approximate 1d median volume per 1h bar by dividing by 16 (approx 16x 1h in 1d)
        vol_median_1h_approx = vol_median_20_1d_aligned[i] / 16.0
        if vol_median_1h_approx <= 0 or np.isnan(vol_median_1h_approx):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_1h_approx * 1.5)
        
        # Camarilla breakout conditions (R3/S3 for continuation)
        breakout_up = curr_close > camarilla_r3_aligned[i]   # break above R3
        breakout_down = curr_close < camarilla_s3_aligned[i] # break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume confirmation
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: Breakout down AND downtrend AND volume confirmation
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakout down (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout up (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals