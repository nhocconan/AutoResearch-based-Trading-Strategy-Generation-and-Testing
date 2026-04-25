#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance. 
Breakouts above R3 or below S3 with 1d EMA34 trend alignment and volume spike 
signal strong momentum moves. Works in bull (breakouts up) and bear (breakouts down) 
markets by following the trend. Targets 75-200 total trades over 4 years.
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
    
    # Get 1d data for EMA34 trend and Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = df_1d['close'] + range_1d * 1.1 / 4
    camarilla_s3 = df_1d['close'] - range_1d * 1.1 / 4
    
    # Align HTF data to LTF (no extra delay for EMA, 1 bar delay for pivots)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Calculate 20-period volume MA for 4h volume confirmation
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and HTF data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        camarilla_r3 = camarilla_r3_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_close > camarilla_r3 and 
                         curr_close > ema_trend and volume_confirm)
            # Short: price breaks below S3 AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_close < camarilla_s3 and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below S3 (reversal) OR price falls below EMA34
            if (curr_close < camarilla_s3 or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above R3 (reversal) OR price rises above EMA34
            if (curr_close > camarilla_r3 or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0