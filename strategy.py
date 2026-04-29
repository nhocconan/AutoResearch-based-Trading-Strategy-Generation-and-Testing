#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2x 20-period average)
# Camarilla R3/S3 levels act as strong intraday support/resistance; breakouts capture momentum with institutional participation.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws in both bull/bear markets.
# Volume spike filter (>2x average) confirms significant market interest, reducing false breakouts.
# Discrete position sizing (0.25) minimizes fee churn while maintaining meaningful exposure.
# Target: 75-150 total trades over 4 years (19-37/year) on 6h timeframe.

name = "6h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Calculate daily pivot points from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = close + range * 1.1/4, S3 = close - range * 1.1/4
    camarilla_r3_1d = close_1d + range_1d * 1.1 / 4
    camarilla_s3_1d = close_1d - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate 20-period average volume for spike confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 1d EMA34, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 2x 20-period average
        vol_spike = curr_volume > 2.0 * curr_vol_ma
        
        # Camarilla breakout conditions
        breakout_long = curr_high > curr_r3   # price breaks above R3
        breakout_short = curr_low < curr_s3   # price breaks below S3
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price retracement to pivot level OR trend turns bearish
            pivot_aligned = align_htf_to_ltf(prices, df_1d, 
                                           (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3)
            if (not np.isnan(pivot_aligned[i]) and 
                (curr_close < pivot_aligned[i] or curr_close < curr_ema_1d)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retracement to pivot level OR trend turns bullish
            pivot_aligned = align_htf_to_ltf(prices, df_1d, 
                                           (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3)
            if (not np.isnan(pivot_aligned[i]) and 
                (curr_close > pivot_aligned[i] or curr_close > curr_ema_1d)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout above R3 AND above 1d EMA34 AND volume spike
            if (breakout_long and 
                curr_close > curr_ema_1d and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish breakout below S3 AND below 1d EMA34 AND volume spike
            elif (breakout_short and 
                  curr_close < curr_ema_1d and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals