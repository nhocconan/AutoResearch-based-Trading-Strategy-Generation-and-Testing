#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for weekly calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly high, low, close using previous 5 trading days (approximation)
    # We use a rolling window of 5 days to get the weekly values
    weekly_high = np.full_like(high_1d, np.nan)
    weekly_low = np.full_like(low_1d, np.nan)
    weekly_close = np.full_like(close_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 5:
            weekly_high[i] = np.max(high_1d[i-5:i])
            weekly_low[i] = np.min(low_1d[i-5:i])
            weekly_close[i] = close_1d[i-1]  # Previous day's close as weekly close
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    weekly_p = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_p - weekly_low
    weekly_s1 = 2 * weekly_p - weekly_high
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_r2 = weekly_p + (weekly_high - weekly_low)
    weekly_s2 = weekly_p - (weekly_high - weekly_low)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_r3 = weekly_high + 2 * (weekly_p - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_p)
    
    # Align weekly pivot levels to 1d timeframe
    wp_aligned = align_htf_to_ltf(prices, df_1d, weekly_p)
    wr1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    ws1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    wr2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    ws2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    wr3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    ws3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 1d
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume spike (volume > 1.8x 20-period average for stricter filter)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.8 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr3_aligned[i]) or
            np.isnan(ws3_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 30-period average
        if i >= 30:
            atr_ma_1d = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
            atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
            vol_filter = atr_1d_aligned[i] > atr_ma_1d_aligned[i] if not np.isnan(atr_ma_1d_aligned[i]) else False
        else:
            vol_filter = False
        
        # Only trade with volume spike and volatility filter
        trade_allowed = volume_spike_1d_aligned[i] and vol_filter
        
        if position == 0:
            # Long: price touches or goes below S3 (strong support) with volume spike
            if trade_allowed and close[i] <= ws3_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R3 (strong resistance) with volume spike
            elif trade_allowed and close[i] >= wr3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches midpoint between S2 and S3 or shows weakness
            midpoint_s2_s3 = (ws2_aligned[i] + ws3_aligned[i]) / 2.0
            if close[i] >= midpoint_s2_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches midpoint between R2 and R3 or shows weakness
            midpoint_r2_r3 = (wr2_aligned[i] + wr3_aligned[i]) / 2.0
            if close[i] <= midpoint_r2_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R3_S3_StrictVolume"
timeframe = "1d"
leverage = 1.0