#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter, volume spike, and choppiness regime filter. 
Targets 75-200 total trades over 4 years by requiring confluence of 1d trend, volume spike, and chop regime (trending only). 
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via 1d trend filter and chop regime to avoid ranging markets.
Primary timeframe: 4h, HTF: 1d for trend and Camarilla calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R1, S1, R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=5, adjust=False, min_periods=5).mean().values  # Camarilla uses 5-period ATR-like smoothing
    
    # Camarilla levels based on previous day's close
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses own close
    
    R1 = prev_close + (1.1/12) * (high_1d - low_1d)
    S1 = prev_close - (1.1/12) * (high_1d - low_1d)
    R3 = prev_close + (1.1/6) * (high_1d - low_1d)
    S3 = prev_close - (1.1/6) * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h (no extra delay needed as they're based on completed 1d candles)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: volume > 2.0x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Choppiness regime filter: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    # Calculate CHOP on 4h data using 14-period
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    highest_high_14 = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low_14 = low_series.rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(atr_14 / chop_denom * np.sqrt(14)) / np.log10(14)
    chop_regime = chop < 38.2  # Trending regime only
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 20 for volume median, 14 for chop
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        regime_trending = chop_regime[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R1 with volume spike, uptrend (close > EMA34_1d), and trending regime
            long_entry = (close_val > R1_aligned[i]) and vol_spike and (close_val > ema_34_val) and regime_trending
            # Short: price breaks below S1 with volume spike, downtrend (close < EMA34_1d), and trending regime
            short_entry = (close_val < S1_aligned[i]) and vol_spike and (close_val < ema_34_val) and regime_trending
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal, price re-enters S3, or chop regime turns ranging
            if (close_val < ema_34_val or 
                close_val < S3_aligned[i] or 
                not regime_trending):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, price re-enters R3, or chop regime turns ranging
            if (close_val > ema_34_val or 
                close_val > R3_aligned[i] or 
                not regime_trending):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0