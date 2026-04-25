#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ChopRegime_v2
Hypothesis: 4-hour Camarilla R3/S3 breakout with 1-day EMA34 trend filter, volume spike (>2.0x 20-period average), and choppiness regime filter (CHOP < 38.2 for trending). Only trade breakouts in trending regimes aligned with 1-day EMA34 direction. Uses ATR trailing stop (2.0*ATR) for exits. Designed for 75-150 total trades over 4 years via tight entry conditions (trend + volume + regime + breakout confluence).
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
    
    # Get 1d data for trend filter and Camarilla calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d close aligned for direct trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels on 1d data (based on previous day's range)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First day: use same day's data
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    daily_range = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + 1.1 * daily_range
    camarilla_s3 = prev_close_1d - 1.1 * daily_range
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # ATR for stoploss (20-period)
    atr_period = 20
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index regime filter (14-period) - trending when CHOP < 38.2
    chop_period = 14
    true_range_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    max_min_range = hh - ll
    # Avoid division by zero and invalid values
    max_min_range = np.where((max_min_range == 0) | np.isnan(max_min_range), 1e-10, max_min_range)
    true_range_sum = np.where(np.isnan(true_range_sum), 0, true_range_sum)
    chop = 100 * np.log10(true_range_sum / (np.log10(chop_period) * max_min_range))
    chop = np.nan_to_num(chop, nan=50.0, posinf=100.0, neginf=0.0)
    trending_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, chop_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_34_1d_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        is_uptrend = close_1d_aligned[i] > ema_trend
        
        if position == 0:
            # Only trade in trending regimes with volume spike and breakout
            if trending_regime[i] and vol_spike[i]:
                if is_uptrend:
                    # Long: break above R3 in uptrend
                    if close[i] > r3:
                        signals[i] = 0.25
                        position = 1
                        long_extreme = close[i]
                else:
                    # Short: break below S3 in downtrend
                    if close[i] < s3:
                        signals[i] = -0.25
                        position = -1
                        short_extreme = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: ATR trailing stop (2.0*ATR) or break below S3
            atr_stop = long_extreme - 2.0 * atr[i]
            if close[i] <= atr_stop or close[i] < s3:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions: ATR trailing stop (2.0*ATR) or break above R3
            atr_stop = short_extreme + 2.0 * atr[i]
            if close[i] >= atr_stop or close[i] > r3:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ChopRegime_v2"
timeframe = "4h"
leverage = 1.0