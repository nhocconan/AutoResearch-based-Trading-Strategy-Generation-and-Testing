#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Uses Camarilla pivot levels from 1d to identify key R3/S3 levels. Breakouts above R3 or below S3
# are filtered by 12h EMA50 trend and volume > 2.0x 20-period median. Works in bull/bear markets.
# Discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA50 and volume median
    start_idx = max(50, 20) + 1  # 51
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Need at least 2 completed 1d bars for Camarilla calculation (today's requires yesterday's data)
        if i < len(prices):
            # Get index of completed 1d bars up to previous bar
            completed_1d_bars = min(len(df_1d), i // 96)  # 96 = 6h bars per 1d (24*60/60/6)
            if completed_1d_bars < 2:
                signals[i] = 0.0
                if position != 0:
                    position = 0
                continue
            
            # Use previous completed 1d bar for Camarilla calculation (no look-ahead)
            prev_1d_idx = completed_1d_bars - 1
            if prev_1d_idx >= len(df_1d):
                signals[i] = 0.0
                if position != 0:
                    position = 0
                continue
                
            prev_high = df_1d['high'].iloc[prev_1d_idx]
            prev_low = df_1d['low'].iloc[prev_1d_idx]
            prev_close = df_1d['close'].iloc[prev_1d_idx]
            
            # Calculate Camarilla pivot levels for today based on yesterday's OHLC
            range_ = prev_high - prev_low
            if range_ <= 0:
                signals[i] = 0.0
                if position != 0:
                    position = 0
                continue
                
            camarilla_r3 = prev_close + range_ * 1.1 / 4
            camarilla_s3 = prev_close - range_ * 1.1 / 4
            camarilla_r4 = prev_close + range_ * 1.1 / 2
            camarilla_s4 = prev_close - range_ * 1.1 / 2
        else:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Camarilla breakout conditions
        breakout_r3 = curr_close > camarilla_r3   # break above R3
        breakdown_s3 = curr_close < camarilla_s3  # break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 AND uptrend AND volume confirmation
            if breakout_r3 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S3 AND downtrend AND volume confirmation
            elif breakdown_s3 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S3 (reversal signal)
            if breakdown_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above R3 (reversal signal)
            if breakout_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals