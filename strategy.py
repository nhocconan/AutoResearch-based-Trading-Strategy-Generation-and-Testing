#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Trade 12h Camarilla R3/S3 breakouts with 1w EMA34 trend filter and volume spike confirmation.
Uses ATR trailing stop (1.5x) and discrete position sizing (0.25) to minimize fee drag.
Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year).
Uses multiple timeframes: 1w for trend, 1d for pivots, 12h for execution.
Focus on BTC/ETH with volume confirmation and trend alignment to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for Camarilla pivots (using previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align HTF indicators to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 2.0x median volume (using 50-period median for stability)
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    # ATR for stop (20-period on 12h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1w EMA (34), volume median (50), 12h ATR (20)
    start_idx = max(34, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(atr_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1w_val = ema_34_1w_aligned[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_20_val = atr_20[i]
        
        if position == 0:
            # Long: break above R3, uptrend (close > EMA34), volume spike
            long_signal = (high_val > camarilla_r3_val) and \
                          (close_val > ema_34_1w_val) and \
                          (volume_val > 2.0 * vol_median_val)
            # Short: break below S3, downtrend (close < EMA34), volume spike
            short_signal = (low_val < camarilla_s3_val) and \
                           (close_val < ema_34_1w_val) and \
                           (volume_val > 2.0 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 1.5 * atr_20_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 1.5 * atr_20_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 1.5 * atr_20_val)
            # Exit: trailing stop hit or trend reversal (close < EMA34) after minimum holding period
            if bars_since_entry >= 3 and ((low_val < long_stop) or (close_val < ema_34_1w_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 1.5 * atr_20_val)
            # Exit: trailing stop hit or trend reversal (close > EMA34) after minimum holding period
            if bars_since_entry >= 3 and ((high_val > short_stop) or (close_val > ema_34_1w_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0