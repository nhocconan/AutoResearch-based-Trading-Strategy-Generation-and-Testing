# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels (R3/S3) from daily timeframe act as strong support/resistance.
# Breakouts above R3 or below S3 with volume confirmation and 1d trend filter capture
# institutional moves. Works in both bull (breakouts continue) and bear (breakdowns continue)
# regimes. Uses 4h timeframe for entries, 1d for pivots/trend/volume confirmation.
# Target: 20-50 trades/year per symbol, low frequency to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1d data for Camarilla pivots, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R3 = close + 1.1*(high-low)*1.1/2, S3 = close - 1.1*(high-low)*1.1/2
    # Using standard Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    camarilla_width = (high_1d - low_1d) * 1.1 / 2
    r3_level = close_1d + camarilla_width  # R3 level
    s3_level = close_1d - camarilla_width  # S3 level
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average for spike detection (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h volume for confirmation
    volume_4h = prices['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 2x 20-period 4h MA
        vol_ok = volume_4h[i] > 2.0 * vol_ma_4h[i]
        
        # 1d volume spike: current 1d volume > 1.5x 20-period 1d MA (use aligned)
        vol_1d_ok = volume_1d[-1] > 1.5 * vol_ma_1d[-1] if len(volume_1d) > 0 else False
        
        if position == 0:
            # Long: break above R3 with 1d uptrend and volume confirmation
            if (prices['close'].iloc[i] > r3_aligned[i] and 
                prices['close'].iloc[i] > ema_1d_aligned[i] and 
                vol_ok and vol_1d_ok):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with 1d downtrend and volume confirmation
            elif (prices['close'].iloc[i] < s3_aligned[i] and 
                  prices['close'].iloc[i] < ema_1d_aligned[i] and 
                  vol_ok and vol_1d_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks back below R3 or trend fails
            if prices['close'].iloc[i] < r3_aligned[i] or prices['close'].iloc[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks back above S3 or trend fails
            if prices['close'].iloc[i] > s3_aligned[i] or prices['close'].iloc[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals