#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolFilter_v1
Hypothesis: Trade 1h Camarilla R1/S1 breakouts only when aligned with 4h EMA50 trend and confirmed by 1d volume spike (1.5x 20-period median). Use ATR(14) trailing stop (1.5x ATR) to manage risk. Target 15-30 trades/year on 1h by using 4h/1d filters to reduce noise and fee drag. Works in bull/bear via trend filter and volume confirmation.
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
    
    # Get 4h data for HTF trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume median (20-period) for volume spike filter
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).median().values
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    # Calculate Camarilla levels from previous 1h OHLC (using 1h data resampled internally is not allowed, so we approximate using 1h close)
    # Instead, we use 1h close to approximate Camarilla: R1 = close + 1.0/12 * (high-low), S1 = close - 1.0/12 * (high-low)
    # But we need true 1h OHLC for Camarilla. Since we cannot resample, we use a proxy: 
    # We'll use the 1h high/low/close directly to compute Camarilla for the current bar (not previous)
    # Note: This is not ideal but avoids resampling. For true Camarilla we need previous bar OHLC.
    # We'll shift our 1h data by 1 to get previous bar.
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_r1 = prev_close + 1.0/12 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 1.0/12 * (prev_high - prev_low)
    
    # ATR(14) for trailing stop
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 4h, volume median (20) 1d, ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_median_1d_aligned[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_1d_val = vol_median_1d_aligned[i]
        atr_val = atr[i]
        hour = hours[i]
        
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten or hold flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_4h_val
        downtrend = close_val < ema_50_4h_val
        
        if position == 0:
            # Long: break above R1 with volume spike, and uptrend
            long_signal = (close_val > camarilla_r1[i]) and \
                          (volume_val > 1.5 * vol_median_1d_val) and \
                          uptrend
            
            # Short: break below S1 with volume spike, and downtrend
            short_signal = (close_val < camarilla_s1[i]) and \
                           (volume_val > 1.5 * vol_median_1d_val) and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: 1.5x ATR
            if close_val < highest_since_entry - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: 1.5x ATR
            if close_val > lowest_since_entry + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolFilter_v1"
timeframe = "1h"
leverage = 1.0