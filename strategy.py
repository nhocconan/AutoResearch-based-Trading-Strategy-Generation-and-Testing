#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTFConfirm_v2
Hypothesis: Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation. 
Uses HTF 12h EMA50 for additional trend alignment to reduce false breakouts. 
Targets 50-150 total trades over 4 years by requiring confluence of trend (1d+12h), volume, and Camarilla breakout. 
Works in bull/bear markets via dual timeframe trend filter (1d EMA50 + 12h EMA50). 
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 12h data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 50 or len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h EMA50 for additional trend confirmation
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d ATR(14) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla pivot levels from previous 1d
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    # We use previous day's high/low/close for today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_r4 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s4 = prev_close - 1.1 * (prev_high - prev_low)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Reduced fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 50 for 12h EMA, 14 for ATR, 20 for volume median
    start_idx = max(50, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R3 with volume spike and both timeframes bullish
            long_entry = (high[i] > camarilla_r3_aligned[i]) and vol_spike and (close_val > ema_50_1d_val) and (close_val > ema_50_12h_val)
            # Short: price breaks below S3 with volume spike and both timeframes bearish
            short_entry = (low[i] < camarilla_s3_aligned[i]) and vol_spike and (close_val < ema_50_1d_val) and (close_val < ema_50_12h_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal in either timeframe, ATR stoploss, or at R4 (take profit)
            stop_price = entry_price - 2.0 * atr_val
            if (close_val < ema_50_1d_val or close_val < ema_50_12h_val or 
                close_val < stop_price or high[i] > camarilla_r4_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal in either timeframe, ATR stoploss, or at S4 (take profit)
            stop_price = entry_price + 2.0 * atr_val
            if (close_val > ema_50_1d_val or close_val > ema_50_12h_val or 
                close_val > stop_price or low[i] < camarilla_s4_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTFConfirm_v2"
timeframe = "6h"
leverage = 1.0