#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses tight entry conditions (breakout of R3/S3 levels) only when aligned with 1d trend and elevated volume.
# Targets 12-25 trades/year by requiring confluence of price structure, trend, and volume.
# Works in bull markets (breakout continuation) and bear markets (breakdown continuation) by following 1d trend.
# Volume spike ensures participation and reduces false breakouts.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA trend and prior day OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior day OHLC for Camarilla levels (using prior day's OHLC for current day's levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day OHLC for current day's Camarilla levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R3, S3, R4, S4
    range_1d = prev_high - prev_low
    r3_1d = prev_close + (range_1d * 1.1 / 4)
    s3_1d = prev_close - (range_1d * 1.1 / 4)
    r4_1d = prev_close + (range_1d * 1.1 / 2)
    s4_1d = prev_close - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume spike: volume > 1.5 * 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_r3 = r3_1d_aligned[i]
        curr_s3 = s3_1d_aligned[i]
        curr_r4 = r4_1d_aligned[i]
        curr_s4 = s4_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Trend filter: price above EMA34 = uptrend, below = downtrend
        is_uptrend = curr_close > curr_ema_34
        is_downtrend = curr_close < curr_ema_34
        
        if position == 0:  # Flat - look for new entries
            # Long: Uptrend AND price breaks above R3 WITH volume spike
            if (is_uptrend and 
                curr_high > curr_r3 and 
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend AND price breaks below S3 WITH volume spike
            elif (is_downtrend and 
                  curr_low < curr_s3 and 
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 (failed breakout) OR closes below EMA34 (trend change)
            if (curr_close < curr_s3 or 
                curr_close < curr_ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 (failed breakdown) OR closes above EMA34 (trend change)
            if (curr_close > curr_r3 or 
                curr_close > curr_ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals