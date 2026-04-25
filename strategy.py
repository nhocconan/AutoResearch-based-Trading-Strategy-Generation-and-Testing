#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above R3 in uptrend (close > 1d EMA50) with volume spike.
Short when price breaks below S3 in downtrend (close < 1d EMA50) with volume spike.
Exit when price re-enters the Camarilla H-L range (H3-L3) or trend reverses.
Designed for low trade frequency (target: 20-50/year) and robustness in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar using previous bar's HLC
    # Camarilla levels based on previous 4h bar's high, low, close
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    # First bar has no previous data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_hl * 1.1 / 2)
    R2 = pivot + (range_hl * 1.1 / 4)
    R1 = pivot + (range_hl * 1.1 / 6)
    S1 = pivot - (range_hl * 1.1 / 6)
    S2 = pivot - (range_hl * 1.1 / 4)
    S3 = pivot - (range_hl * 1.1 / 2)
    H3 = R3  # Same as R3
    L3 = S3  # Same as S3
    
    # Align Camarilla levels to original timeframe
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    H3_aligned = align_htf_to_ltf(prices, df_4h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_4h, L3)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (daily)
                # Long: break above R3 with volume spike
                long_signal = (close[i] > R3_aligned[i]) and vol_spike[i]
                # Short: break below S3 only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < S3_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (daily)
                # Short: break below S3 with volume spike
                short_signal = (close[i] < S3_aligned[i]) and vol_spike[i]
                # Long: break above R3 only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > R3_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: re-enter H-L range (H3-L3) or trend reversal
            exit_signal = (close[i] < H3_aligned[i]) or (close[i] > L3_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter H-L range (H3-L3) or trend reversal
            exit_signal = (close[i] > L3_aligned[i]) or (close[i] < H3_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0