#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Uses 1d Camarilla pivot levels (R1/S1) for breakout entries with 1d EMA34 trend filter, volume confirmation, and 12h choppiness regime filter to avoid whipsaws in sideways markets. Long when price breaks above R1 AND 1d close > EMA34 (uptrend) AND volume > 2.0 * 20-period average AND chop < 61.8 (trending regime). Short when price breaks below S1 AND 1d close < EMA34 (downtrend) AND volume > 2.0 * 20-period average AND chop < 61.8. Exit when price returns to the pivot level (R1 for longs, S1 for shorts) OR trend reverses OR chop > 61.8 (range regime). Designed for 12h timeframe to achieve 50-150 total trades over 4 years with low fee drag. Works in both bull and bear markets by following 1d trend while using Camarilla levels for precise breakout entries and avoiding false signals in choppy markets.
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
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels: R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # 12h Choppiness Index regime filter: CHOP < 61.8 = trending (favor breakouts), CHOP > 61.8 = ranging (avoid breakouts)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, N) - min(low, N))) / log10(N)
    # Using ATR(14) and 14-period lookback
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Avoid division by zero
    atr_safe = np.where(atr_14 == 0, 1e-10, atr_14)
    sum_atr_14 = pd.Series(atr_safe).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero
    range_safe = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10(sum_atr_14 / range_safe) / np.log10(14)
    chop_filter = chop < 61.8  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20), chop (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_34_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R1/S1 with 1d trend filter AND volume AND trending regime
            # Long: price breaks above R1 (minor resistance) AND 1d uptrend AND volume AND chop < 61.8
            long_condition = (close_val > r1_level) and (close_val > ema_val) and vol_conf and chop_ok
            # Short: price breaks below S1 (minor support) AND 1d downtrend AND volume AND chop < 61.8
            short_condition = (close_val < s1_level) and (close_val < ema_val) and vol_conf and chop_ok
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to R1 level OR trend breaks OR chop > 61.8 (range regime)
            exit_condition = (close_val <= r1_level) or (close_val < ema_val) or (not chop_ok)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to S1 level OR trend breaks OR chop > 61.8 (range regime)
            exit_condition = (close_val >= s1_level) or (close_val > ema_val) or (not chop_ok)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0