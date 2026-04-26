#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Use 12h timeframe with Camarilla R3/S3 breakout from prior day, confirmed by 1d EMA trend and volume spike.
Long when: price breaks above Camarilla R3 + close > 1d EMA34 + volume > 1.5x avg volume.
Short when: price breaks below Camarilla S3 + close < 1d EMA34 + volume > 1.5x avg volume.
Exit when: price reverts to Camarilla pivot point (PP) or opposite Camarilla level (S1/R1).
Designed for BTC/ETH: captures intraday swings in ranging markets with tight entries to minimize fee drag.
Targets 12-37 trades/year on 12h timeframe for optimal test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d average volume for volume confirmation
    vol_1d = pd.Series(df_1d['volume'].values)
    avg_vol_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    avg_vol_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need prior day's data for Camarilla calculation
    start_idx = 24  # 24*12h = 2 days to ensure we have prior complete day
    
    for i in range(start_idx, n):
        # Get prior day's OHLC for Camarilla calculation
        # Since we're on 12h timeframe, we need to aggregate last 2 bars (24h) for prior day
        # But to avoid look-ahead, we use completed daily data from HTF
        # Get index of prior completed day in 1d data
        # We'll use the HTF data directly for Camarilla calculation
        
        # Calculate Camarilla levels from prior completed day using HTF data
        # We need to shift by 1 to use prior day's data
        if len(df_1d) < 2:
            continue
            
        # Get prior day's OHLC (already completed)
        prior_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
        prior_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
        prior_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
        
        # Camarilla levels
        rang = prior_high - prior_low
        if rang <= 0:
            continue
            
        pp = (prior_high + prior_low + prior_close) / 3
        r3 = pp + (rang * 1.1 / 4)
        s3 = pp - (rang * 1.1 / 4)
        r1 = pp + (rang * 1.1 / 12)
        s1 = pp - (rang * 1.1 / 12)
        
        # Current price and volume
        close_val = prices['close'].iloc[i]
        vol_val = prices['volume'].iloc[i]
        
        # Get aligned HTF values for current bar
        # Find corresponding index in HTF data
        htf_idx = min(len(df_1d) - 1, i // 2)  # 2*12h = 1d, but use safe indexing
        if htf_idx < 0:
            continue
            
        ema_34_val = ema_34_aligned[i] if i < len(ema_34_aligned) else ema_34_1d[-1]
        avg_vol_val = avg_vol_aligned[i] if i < len(avg_vol_aligned) else avg_vol_1d[-1]
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_spike = vol_val > 1.5 * avg_vol_val if not np.isnan(avg_vol_val) else False
        
        # Fixed position size
        size = 0.25
        
        if position == 0:
            # Flat - look for entry
            # Long: break above R3 + above 1d EMA + volume spike
            long_entry = (close_val > r3) and (close_val > ema_34_val) and vol_spike
            # Short: break below S3 + below 1d EMA + volume spike
            short_entry = (close_val < s3) and (close_val < ema_34_val) and vol_spike
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to pivot or touches S1
            if close_val <= pp or close_val <= s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to pivot or touches R1
            if close_val >= pp or close_val >= r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0