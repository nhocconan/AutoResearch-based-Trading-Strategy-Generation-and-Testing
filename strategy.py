#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w trend filter and volume spike confirmation.
# Uses Camarilla pivot levels from 1d data to identify R3/S3 levels.
# Trend filter: 1w EMA50 direction (bullish if price > EMA50, bearish if price < EMA50).
# Entry: Long when price breaks above R3 with uptrend and volume > 2x 20-period median volume.
# Entry: Short when price breaks below S3 with downtrend and volume > 2x 20-period median volume.
# Exit: Reverse signal from opposite Camarilla level (R4 for long exit, S4 for short exit).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "6h_Camarilla_R3S3_Breakout_1wEMA50_VolumeConfirm_v1"
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
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for volume median
    start_idx = 20  # volume median needs 20 bars
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate Camarilla levels from previous 1d bar (OHLC from df_1d)
        # We need the previous completed 1d bar's OHLC
        # Find index of the 1d bar that corresponds to current time
        # Since we're using 6h bars, each 1d bar = 4 of our 6h bars
        # But we must use align_htf_to_ltf for proper timing
        
        # Get previous 1d bar's OHLC (completed bar)
        # We'll use shift(1) on the 1d data to get previous bar's values
        if len(df_1d) >= 2:
            prev_1d_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
            prev_1d_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
            prev_1d_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
            prev_1d_open = df_1d['open'].iloc[-2] if len(df_1d) >= 2 else df_1d['open'].iloc[-1]
        else:
            # Not enough 1d data yet
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        # Calculate Camarilla levels for R3, S3, R4, S4
        # Camarilla formulas:
        # R4 = close + ((high - low) * 1.1/2)
        # R3 = close + ((high - low) * 1.1/4)
        # S3 = close - ((high - low) * 1.1/4)
        # S4 = close - ((high - low) * 1.1/2)
        rng = prev_1d_high - prev_1d_low
        if rng <= 0:
            r3 = s3 = r4 = s4 = prev_1d_close
        else:
            r3 = prev_1d_close + (rng * 1.1 / 4)
            s3 = prev_1d_close - (rng * 1.1 / 4)
            r4 = prev_1d_close + (rng * 1.1 / 2)
            s4 = prev_1d_close - (rng * 1.1 / 2)
        
        # Trend filter: 1w EMA50 direction
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Breakout conditions
        breakout_r3 = curr_close > r3   # break above R3
        breakout_s3 = curr_close < s3   # break below S3
        breakout_r4 = curr_close > r4   # break above R4 (exit long)
        breakout_s4 = curr_close < s4   # break below S4 (exit short)
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 AND uptrend AND volume confirmation
            if breakout_r3 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S3 AND downtrend AND volume confirmation
            elif breakout_s3 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below S4 (strong reversal) or break above R4 (continuation - but we exit to avoid whipsaw)
            if breakout_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above R4 (strong reversal)
            if breakout_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals