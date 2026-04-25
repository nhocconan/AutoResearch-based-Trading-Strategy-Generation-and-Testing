#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance. 
Breakout above R3 or below S3 with 1d EMA34 trend alignment and volume spike 
signals institutional participation. Works in bull/bear via trend filter.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's Camarilla levels (using 1d OHLC)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2 where C=(H+L+O)/3
    # We need previous day's OHLC, so we shift the 1d data by 1
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # Calculate Camarilla R3 and S3 for previous day
    pivot = (prev_high + prev_low + prev_close) / 3
    rang = prev_high - prev_low
    r3 = pivot + (rang * 1.1 / 2)
    s3 = pivot - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (1d -> 4h)
    # Need extra delay because Camarilla uses previous day's data
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    
    # Calculate 20-period volume MA for 4h volume confirmation
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and Camarilla
    start_idx = max(34, 20) + 1  # +1 for previous day shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_4h
        
        # Breakout conditions
        breakout_long = curr_close > r3_val
        breakout_short = curr_close < s3_val
        
        if position == 0:
            # Look for entry signals
            # Long: Breakout above R3 AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = breakout_long and (curr_close > ema_trend) and volume_confirm
            # Short: Breakout below S3 AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = breakout_short and (curr_close < ema_trend) and volume_confirm
            
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
            # Exit: Break below S3 (reversal) OR price falls below EMA34
            if (curr_close < s3_val) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Break above R3 (reversal) OR price rises above EMA34
            if (curr_close > r3_val) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0