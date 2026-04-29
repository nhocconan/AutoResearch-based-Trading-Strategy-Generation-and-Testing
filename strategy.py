#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike (>2x 20-period average)
# Camarilla R3/S3 levels act as strong intraday support/resistance; breakouts capture momentum with institutional participation.
# 4h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws in both bull/bear markets.
# Volume spike filter (>2x average) confirms significant market interest, reducing false breakouts.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Discrete position sizing (0.20) minimizes fee churn while maintaining meaningful exposure.
# Target: 60-150 total trades over 4 years = 15-37/year on 1h timeframe.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
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
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily pivot points from 4h OHLC (Camarilla needs daily range)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Camarilla levels: R3 = close + range * 1.1/4, S3 = close - range * 1.1/4
    camarilla_r3_4h = close_4h + range_4h * 1.1 / 4
    camarilla_s3_4h = close_4h - range_4h * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Calculate 20-period average volume for spike confirmation (on 1h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 4h EMA50, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_4h = ema_50_4h_aligned[i]
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
            pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
            if (not np.isnan(pivot_aligned[i]) and 
                (curr_close < pivot_aligned[i] or curr_close < curr_ema_4h)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price retracement to pivot level OR trend turns bullish
            pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
            if (not np.isnan(pivot_aligned[i]) and 
                (curr_close > pivot_aligned[i] or curr_close > curr_ema_4h)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout above R3 AND above 4h EMA50 AND volume spike
            if (breakout_long and 
                curr_close > curr_ema_4h and 
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Short entry: bearish breakout below S3 AND below 4h EMA50 AND volume spike
            elif (breakout_short and 
                  curr_close < curr_ema_4h and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals