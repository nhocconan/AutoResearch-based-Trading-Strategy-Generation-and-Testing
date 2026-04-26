#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_Volume_Breakout_Trend_v1
Hypothesis: Trade Camarilla pivot breakouts on 1h with 4h/1d trend and volume confirmation.
Camarilla levels (R1,R2,S1,S2) act as intraday support/resistance. Breakouts with volume
and aligned 4h/1d trend have high win rate. Designed for low trade frequency (15-37/year)
on 1h to minimize fee drag. Uses discrete position sizing (0.20) and session filter (08-20 UTC)
to reduce noise. Works in bull/bear markets by following 4h/1d EMA50 trend.
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
    
    # Get 4h and 1d data for HTF trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA(50) and 1d EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivots using previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # We use R1, R2, S1, S2: R1 = close + 1.125*(high-low), R2 = close + 1.25*(high-low)
    # S1 = close - 1.125*(high-low), S2 = close - 1.25*(high-low)
    # Need previous day's OHLC - we'll approximate using rolling window on 1d data
    # For 1h timeframe, we use the previous 1d candle's OHLC
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get previous day's OHLC for Camarilla calculation (using 1d data)
    # We need to shift the 1d data by 1 to get previous day's values
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels from previous 1d OHLC
    camarilla_r1 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_r2 = prev_close_1d + 1.250 * (prev_high_1d - prev_low_1d)
    camarilla_s1 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_s2 = prev_close_1d - 1.250 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 1h timeframe (they change only at 1d boundaries)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Volume confirmation: 1.5x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 4h EMA (50), 1d EMA (50), volume median (20), Camarilla needs 1d data
    start_idx = max(50, 50, 20)  # EMA50 on 4h/1d, vol median 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_r2_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_s2_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        hour = hours[i]
        
        # Session filter: 08-20 UTC
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten or hold flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with volume and uptrend (close > 4h EMA50 AND close > 1d EMA50)
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 1.5 * vol_median_val) and \
                          (close_val > ema_50_4h_val) and \
                          (close_val > ema_50_1d_val)
            
            # Short: break below S1 with volume and downtrend (close < 4h EMA50 AND close < 1d EMA50)
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 1.5 * vol_median_val) and \
                           (close_val < ema_50_4h_val) and \
                           (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price breaks below S1 (reversal) or trend changes (close < 4h EMA50 OR close < 1d EMA50)
            if (close_val < camarilla_s1_aligned[i]) or \
               (close_val < ema_50_4h_val) or \
               (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 (reversal) or trend changes (close > 4h EMA50 OR close > 1d EMA50)
            if (close_val > camarilla_r1_aligned[i]) or \
               (close_val > ema_50_4h_val) or \
               (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_Pivot_Volume_Breakout_Trend_v1"
timeframe = "1h"
leverage = 1.0