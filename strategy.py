#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>1.8x 20-period average)
# Uses proven Camarilla pivot structure for precise entry/exit levels. 1d EMA34 ensures higher timeframe trend alignment.
# Volume spike filters for institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Effective in bull markets (breakouts) and bear markets (mean reversion at extremes).
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    # Camarilla: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d_vals, 1)
    prev_high_1d[0] = np.nan  # First day has no previous
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels for previous day
    camarilla_r3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 1d EMA34 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = curr_volume > 1.8 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR trend turns bearish (price below 1d EMA34)
            if curr_close < curr_s3 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend turns bullish (price above 1d EMA34)
            if curr_close > curr_r3 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND above 1d EMA34 AND volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND below 1d EMA34 AND volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals