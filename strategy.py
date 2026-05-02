#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 1d HTF for EMA34 to capture daily trend and reduce false breakouts in choppy markets.
# Camarilla R3/S3 from 6h provides proven intraday reversal/continuation levels with good historical performance.
# Volume confirmation at 2.0x average ensures strong participation while limiting trades (~12-37/year target).
# Discrete sizing 0.25 to minimize fee churn. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Volume"
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
    
    # Calculate Camarilla levels R3 and S3 from 6h timeframe (using prior completed 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Prior 6h bar's high, low, close for Camarilla calculation
    prev_high_6h = df_6h['high'].shift(1).values
    prev_low_6h = df_6h['low'].shift(1).values
    prev_close_6h = df_6h['close'].shift(1).values
    
    # Camarilla R3 and S3 levels (proven breakout/continuation levels)
    camarilla_r3_6h = prev_close_6h + (prev_high_6h - prev_low_6h) * 1.1 / 4
    camarilla_s3_6h = prev_close_6h - (prev_high_6h - prev_low_6h) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (they are already 6h, but align for safety)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3_6h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3_6h)
    
    # 1d EMA34 for trend filter (daily trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 AND price > 1d EMA34 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND price < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below S3 OR price < 1d EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above R3 OR price > 1d EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals