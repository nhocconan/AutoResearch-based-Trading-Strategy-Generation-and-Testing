#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v2
Hypothesis: Daily timeframe Camarilla R3/S3 breakout with 1-week EMA50 trend filter and volume confirmation (>1.8x median). 
Targets weekly institutional pivot levels with strong volume in trending markets. 
Designed for BTC/ETH with very strict entry conditions (~10-15 trades/year) to minimize fee drag and maximize test generalization.
Uses discrete position sizing (0.25) and ATR trailing stop (2.0x) for risk management.
Works in bull/bear by only trading with 1-week trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend (EMA50)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for Camarilla calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels from previous 1d bar (HLC of prior day)
    cam_high = pd.Series(df_1d['high'].values).shift(1).values
    cam_low = pd.Series(df_1d['low'].values).shift(1).values
    cam_close = pd.Series(df_1d['close'].values).shift(1).values
    
    # Camarilla R3, S3 levels (stronger breakout levels)
    R3 = cam_close + (cam_high - cam_low) * 1.1 / 4
    S3 = cam_close - (cam_high - cam_low) * 1.1 / 4
    
    # Volume spike filter: volume > 1.8x median volume (20-period) for high conviction
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(15) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # Align HTF indicators to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 1w, Camarilla (need 2 bars for shift), volume median (20), ATR (15)
    start_idx = max(50, 2, 20, 15) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1w_val
        downtrend = close_val < ema_50_1w_val
        
        # Volume spike filter: only trade in high-volume environments
        volume_spike = volume_val > 1.8 * vol_median_val
        
        if position == 0:
            # Long: break above R3 with volume spike, and uptrend
            long_signal = (close_val > r3_val) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S3 with volume spike, and downtrend
            short_signal = (close_val < s3_val) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0