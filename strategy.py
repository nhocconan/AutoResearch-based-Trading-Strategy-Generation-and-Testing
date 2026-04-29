#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when: price breaks above R3 (Camarilla resistance) AND close > 1d EMA34 (uptrend) AND volume > 1.5 * avg_volume(20)
# Short when: price breaks below S3 (Camarilla support) AND close < 1d EMA34 (downtrend) AND volume > 1.5 * avg_volume(20)
# Uses Camarilla for institutional price levels, EMA for trend filter, volume for conviction, discrete sizing (0.25) to minimize fee churn.
# Works in bull/bear via EMA trend filter (avoids counter-trend trades) + Camarilla levels (mean reversion at extremes).
# Timeframe: 4h (primary), HTF: 1d for EMA34 and Camarilla calculation.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (using prior day's high/low/close)
    # Camarilla: P = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_p = (high_1d + low_1d + close_1d_arr) / 3.0
    camarilla_r3 = close_1d_arr + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3 = close_1d_arr - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below R3 (mean reversion)
            # 2. Close falls below 1d EMA34 (trend change)
            if (curr_close < curr_r3) or (curr_close < curr_ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above S3 (mean reversion)
            # 2. Close rises above 1d EMA34 (trend change)
            if (curr_close > curr_s3) or (curr_close > curr_ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND close > 1d EMA34 AND volume spike
            if (curr_close > curr_r3) and (curr_close > curr_ema_34) and curr_volume_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND close < 1d EMA34 AND volume spike
            elif (curr_close < curr_s3) and (curr_close < curr_ema_34) and curr_volume_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals