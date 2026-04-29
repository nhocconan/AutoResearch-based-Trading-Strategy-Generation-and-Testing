#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivots calculated from previous 1d OHLC: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
# Long: Close > R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short: Close < S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Works in bull via breakout continuation, in bear via mean reversion at extremes (pivots act as support/resistance)
# Uses 12h primary timeframe with 1d HTF for pivots and trend filter to minimize overtrading and fee drag

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels from previous 1d candle
        idx_1d = len(df_1d) - 1  # last completed 1d bar
        if idx_1d < 0:
            signals[i] = 0.0
            continue
            
        prev_1d_high = df_1d['high'].iloc[idx_1d]
        prev_1d_low = df_1d['low'].iloc[idx_1d]
        prev_1d_close = df_1d['close'].iloc[idx_1d]
        
        # Camarilla R3 and S3
        rang = prev_1d_high - prev_1d_low
        r3 = prev_1d_close + 1.1 * rang * 1.1 / 4
        s3 = prev_1d_close - 1.1 * rang * 1.1 / 4
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Close below R3 (breakout failed) OR price below 1d EMA34 (trend change)
            if curr_close < r3 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Close above S3 (breakdown failed) OR price above 1d EMA34 (trend change)
            if curr_close > s3 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Close > R3 AND price > 1d EMA34 AND volume spike
            if (curr_close > r3 and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Close < S3 AND price < 1d EMA34 AND volume spike
            elif (curr_close < s3 and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals