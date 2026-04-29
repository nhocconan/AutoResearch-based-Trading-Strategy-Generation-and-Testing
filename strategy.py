#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above R3 AND 1w EMA34 uptrend AND volume spike (>2x 20-bar avg)
# Short when price breaks below S3 AND 1w EMA34 downtrend AND volume spike
# Exit when price reverts to mean (Pivot Point) OR trend changes
# Camarilla levels provide precise intraday support/resistance,
# 1w EMA34 filters higher timeframe trend,
# volume confirmation ensures momentum validity
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d OHLC for Camarilla levels (yesterday's daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    # R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, R2 = close + (high-low)*1.1/6
    # R1 = close + (high-low)*1.1/12, PP = (high+low+close)/3
    # S1 = close - (high-low)*1.1/12, S2 = close - (high-low)*1.1/6, S3 = close - (high-low)*1.1/4
    # S4 = close - (high-low)*1.1/2
    
    # We need previous day's OHLC, so shift by 1
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = pp + rang * 1.1 / 4.0
    s3 = pp - rang * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_pp = pp_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price below Pivot Point OR trend change (price below 1w EMA34)
            if curr_close < curr_pp or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Pivot Point OR trend change (price above 1w EMA34)
            if curr_close > curr_pp or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price above 1w EMA34 AND volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_1w and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND price below 1w EMA34 AND volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_1w and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals