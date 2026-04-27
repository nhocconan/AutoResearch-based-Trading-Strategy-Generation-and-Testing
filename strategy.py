#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 4h with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above R3 with 12h uptrend and volume spike. Short when price breaks below S3 with 12h downtrend and volume spike.
Camarilla levels provide high-probability reversal/breakout points. 12h EMA50 filters for trend alignment to avoid counter-trend trades.
Volume spike confirms institutional participation. Designed for 20-50 trades/year on 4h to minimize fee drag.
Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).
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
    
    # Calculate Camarilla levels for 4h (based on previous day's OHLC)
    # We need daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # But we use the standard Camarilla formula based on previous day's OHLC
    # Since we don't have previous day's data directly, we'll approximate using rolling window
    # Better approach: use daily data to calculate levels, then align to 4h
    
    # Calculate typical Camarilla levels from daily data
    # R3 = C + 1.1*(H-L), S3 = C - 1.1*(H-L) where C,H,L are daily close,high,low
    # Actually standard Camarilla: R3 = C + 1.1*(H-L), S3 = C - 1.1*(H-L)
    # But we need to use previous day's values, so we shift by 1
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Camarilla R3 and S3 from previous day
    camarilla_r3 = d_close + 1.1 * (d_high - d_low)
    camarilla_s3 = d_close - 1.1 * (d_high - d_low)
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Camarilla calculation (need at least 1 day), EMA50, volume average
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with 12h trend alignment and volume spike
            # Long: price breaks above R3 AND 12h trend up (close > EMA50) AND volume spike
            # Short: price breaks below S3 AND 12h trend down (close < EMA50) AND volume spike
            long_condition = close_val > r3_level and close_val > ema_trend and vol_spike
            short_condition = close_val < s3_level and close_val < ema_trend and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below S3 (reversal) OR 12h trend turns down
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R3 (reversal) OR 12h trend turns up
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0