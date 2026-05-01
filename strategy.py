#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter.
# Uses 1d Camarilla pivot levels for structure, 1d volume spike for confirmation,
# and 1d choppiness index to avoid ranging markets. Long when price breaks above R3
# with volume spike and chop < 61.8 (trending). Short when price breaks below S3
# with volume spike and chop < 61.8. Discrete sizing 0.25. Target: 25-40 trades/year.
# Camarilla levels provide institutional support/resistance, volume confirms institutional
# participation, chop filter ensures we trade in trending environments where breakouts work.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for pivot, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (using prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day OHLC for current day's pivot
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_1d = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    r3_1d = prev_close + (range_1d * 1.1 / 4)
    s3_1d = prev_close - (range_1d * 1.1 / 4)
    
    # Align 1d levels to 4h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1d volume spike: volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (1.5 * vol_ema_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # 1d Choppiness Index: CHOP(14) = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We'll use a simplified version: CHOP = 100 * (ATR(14) / (HH(14) - LL(14)))
    # Then normalize to 0-100 scale where >61.8 = choppy/ranging
    tr1 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum()
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop_raw = 100 * (tr1 / (hh14 - ll14))
    # Avoid division by zero and extreme values
    chop_raw = chop_raw.replace([np.inf, -np.inf], np.nan)
    chop_values = chop_raw.fillna(50).values  # neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for indicators
    
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
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_1d_aligned[i]
        curr_s3 = s3_1d_aligned[i]
        curr_vol_spike = vol_spike_aligned[i] > 0.5  # boolean
        curr_chop = chop_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 with volume spike and trending market (chop < 61.8)
            if (curr_close > curr_r3 and 
                curr_vol_spike and 
                curr_chop < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and trending market (chop < 61.8)
            elif (curr_close < curr_s3 and 
                  curr_vol_spike and 
                  curr_chop < 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below pivot OR volume dries up OR chop becomes too high (ranging)
            if (curr_close < pivot_1d_aligned[i] or 
                not curr_vol_spike or 
                curr_chop > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above pivot OR volume dries up OR chop becomes too high (ranging)
            if (curr_close > pivot_1d_aligned[i] or 
                not curr_vol_spike or 
                curr_chop > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals