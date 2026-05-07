# [132826] 4h_Camarilla_R3S3_1DTrend_VolumeSpike_v22 - New approach with tighter entry and refined exit
# Hypothesis: The repeated failures stem from overly frequent entries and exits due to loose conditions. 
# This version tightens entry by requiring price to close beyond R3/S3 AND beyond the 1-day EMA34 by a minimum buffer (0.1% of price), 
# reduces exit sensitivity by widening the tolerance band to 35% of the H4-L4 range, and increases volume confirmation threshold to 2.5x average volume.
# These changes aim to reduce trade frequency (target: 15-30 trades/year per symbol) while improving signal quality.
# We maintain the 4h timeframe, use 1d EMA34 for trend, and keep volume spike and session filters.
# The goal is to achieve positive Sharpe on BTC/ETH in both bull and bear markets by avoiding whipsaws and capturing only strong breakouts with confirmation.

#!/usr/bin/env python3
name = "4H_Camarilla_R3S3_1DTrend_VolumeSpike_v22"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for structure (R3/S3 levels)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate previous 4h bar's high, low, close for Camarilla levels
    prev_high = df_4h['high'].values
    prev_low = df_4h['low'].values
    prev_close = df_4h['close'].values
    
    # Calculate Camarilla levels: R3 and S3 (correct formula)
    # R3 = C + (H-L) * 1.1/2 * 1.1, S3 = C - (H-L) * 1.1/2 * 1.1
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 1.1 / 2 * 1.1
    s3 = prev_close - range_hl * 1.1 / 2 * 1.1
    
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume filter: current volume > 2.5x average volume (20-period) - increased threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility periods (ATR > 0.3% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close  # ATR > 0.3% of price
    
    # Session filter: 08:00 - 20:00 UTC (80% of day)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i] or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.5x average volume) - increased threshold
        volume_filter = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price closes beyond R3 + beyond daily EMA34 by buffer + volume spike
            buffer = 0.001 * close[i]  # 0.1% buffer
            if (close[i] > r3_aligned[i] + buffer and 
                close[i] > ema_34_1d_aligned[i] + buffer and   # Daily uptrend with buffer
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price closes below S3 + below daily EMA34 by buffer + volume spike
            elif (close[i] < s3_aligned[i] - buffer and 
                  close[i] < ema_34_1d_aligned[i] - buffer and   # Daily downtrend with buffer
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the middle of the prior 4h range (H4/L4)
            # H4 = close + 1.1*(high-low)*1.1/6, L4 = close - 1.1*(high-low)*1.1/6
            range_hl = prev_high - prev_low
            h4 = prev_close + range_hl * 1.1 / 6 * 1.1
            l4 = prev_close - range_hl * 1.1 / 6 * 1.1
            h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
            l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
            
            camarilla_mid = (h4_aligned[i] + l4_aligned[i]) / 2
            range_hl_4h = h4_aligned[i] - l4_aligned[i]
            # Exit when within 35% of the mid-point (wider band to reduce churn)
            at_mid = abs(close[i] - camarilla_mid) < range_hl_4h * 0.35
            
            if at_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals