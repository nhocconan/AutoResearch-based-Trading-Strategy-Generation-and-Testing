#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Volume spike filter: current volume > 2.5x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.5 * vol_ma24)
    
    # Daily data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous day)
    range_1d = high_1d - low_1d
    close_prev = close_1d
    
    # R3, S3 levels
    r3 = close_prev + (range_1d * 1.1 / 4)
    s3 = close_prev - (range_1d * 1.1 / 4)
    # R4, S4 levels (breakout zones)
    r4 = close_prev + (range_1d * 1.1 / 2)
    s4 = close_prev - (range_1d * 1.1 / 2)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d data to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price above R4 with volume spike and uptrend
            long_breakout = (close[i] > r4_aligned[i]) and volume_spike[i] and (close[i] > ema_34_1d_aligned[i])
            # Short breakdown: price below S4 with volume spike and downtrend
            short_breakdown = (close[i] < s4_aligned[i]) and volume_spike[i] and (close[i] < ema_34_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakdown:
                signals[i] = -0.25
                position = -1
            # Fade at R3/S3 in ranging markets (when price is near extremes but no breakout)
            elif (close[i] >= r3_aligned[i]) and (close[i] <= r4_aligned[i]) and volume_spike[i]:
                # Fade long from R3 when in downtrend
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif (close[i] <= s3_aligned[i]) and (close[i] >= s4_aligned[i]) and volume_spike[i]:
                # Fade short from S3 when in uptrend
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Long exit: price crosses below R3 (profit taking or reversal)
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above S3 (profit taking or reversal)
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals