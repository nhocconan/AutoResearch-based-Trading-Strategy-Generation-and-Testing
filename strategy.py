# 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
# Hypothesis: Uses Camarilla pivot levels from 1-day timeframe for breakout trading.
# Long when price breaks above R3 with volume spike and above 1-day EMA34 (uptrend).
# Short when price breaks below S3 with volume spike and below 1-day EMA34 (downtrend).
# Designed for medium trade frequency with clear structure and trend alignment.
# Works in both bull and bear markets by following the intermediate-term trend via EMA filter.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Calculate Camarilla Pivot Levels from 1d data ---
    # Typical price for the day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot point
    pivot = typical_price.values
    # Range
    range_ = df_1d['high'] - df_1d['low']
    # Camarilla levels
    r3 = pivot + 1.1 * range_ * 1.1 / 4  # R3 = pivot + 1.1 * (high-low) * 1.1/4
    s3 = pivot - 1.1 * range_ * 1.1 / 4  # S3 = pivot - 1.1 * (high-low) * 1.1/4
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- 1-day Trend Filter (EMA34 on 1d close) ---
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R3 with volume, above 1-day EMA34
            if (close[i] > r3_aligned[i] and 
                volume_spike and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, below 1-day EMA34
            elif (close[i] < s3_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of trend
            if position == 1:
                # Exit long: price breaks below S3 or loses uptrend
                if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R3 or loses downtrend
                if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals