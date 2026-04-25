#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance. Breakouts above R3 or below S3 with volume confirmation and 1d EMA34 trend alignment capture institutional momentum. Works in both bull (long breakouts) and bear (short breakouts) by taking directional signals. Designed for 4h with tight entry conditions to achieve 20-50 trades/year, minimizing fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    # Camarilla formulas
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r3 = close + range_ * 1.1 / 4.0
    r4 = close + range_ * 1.1 / 2.0
    s3 = close - range_ * 1.1 / 4.0
    s4 = close - range_ * 1.1 / 2.0
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) on 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        r3, r4, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_r3[i] = r3
        camarilla_s3[i] = s3
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = calculate_ema(close_1d, 34)
    
    # Align 1d indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Camarilla calculation (1d needs 1 bar), EMA, volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        camarilla_r3 = camarilla_r3_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Camarilla R3 AND volume spike AND price > EMA34 (uptrend)
            long_entry = (curr_close > camarilla_r3) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Camarilla S3 AND volume spike AND price < EMA34 (downtrend)
            short_entry = (curr_close < camarilla_s3) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Camarilla S3 OR price crosses below EMA34 (trend change)
            if (curr_close < camarilla_s3) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price crosses above Camarilla R3 OR price crosses above EMA34 (trend change)
            if (curr_close > camarilla_r3) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0