#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot breakout with 12-hour trend filter and volume spike confirmation.
Trades long when price breaks above R3 level with volume > 2x 12h average and 12h trend up.
Trades short when price breaks below S3 level with volume > 2x 12h average and 12h trend down.
Uses tight entry conditions (3+ confluence) to target 20-40 trades/year per symbol, minimizing fee drag.
Works in bull/bear markets via 12h trend filter and volume confirmation of breakout strength.
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
    
    # Get 4-hour data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate previous 4-hour Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Shift by 1 to use previous bar's data (no look-ahead)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    # Camarilla levels: H-L range from previous bar
    rang = prev_high_4h - prev_low_4h
    r3 = prev_close_4h + 1.1 * rang / 2
    r4 = prev_close_4h + 1.1 * rang
    s3 = prev_close_4h - 1.1 * rang / 2
    s4 = prev_close_4h - 1.1 * rang
    
    # Align Camarilla levels to 4-hour timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_4h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_4h, s4)
    
    # Get 12-hour data for trend filter and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(30) for trend
    close_12h = df_12h['close'].values
    ema_30_12h = pd.Series(close_12h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # Calculate 12-hour volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Camarilla (shifted), 12h EMA, and 12h volume avg
    start_idx = max(30, 30, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_30_12h_aligned[i]) or 
            np.isnan(vol_avg_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 4-hour price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_avg = vol_avg_20_12h_aligned[i]
        trend_12h = ema_30_12h_aligned[i]
        
        # Volume filter: volume > 2x 12-hour average
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Entry conditions: Camarilla breakout with volume and 12h trend alignment
        if position == 0:
            # Long: breakout above R3 + volume + 12h uptrend
            if price_now > r3_aligned[i] and vol_filter and price_now > trend_12h:
                signals[i] = size
                position = 1
            # Short: breakout below S3 + volume + 12h downtrend
            elif price_now < s3_aligned[i] and vol_filter and price_now < trend_12h:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to S3 level or 12h trend turns down
            if price_now < s3_aligned[i] or price_now < trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to R3 level or 12h trend turns up
            if price_now > r3_aligned[i] or price_now > trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0