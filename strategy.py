#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_RegimeFilter
Hypothesis: On 4h timeframe, Camarilla R3/S3 breakouts with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trending) capture high-probability institutional breakouts. Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear via trend filter + regime adaptation.
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
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for choppiness regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR(14) for 1d
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate Choppiness Index (CHOP) for 1d
    # CHOP = 100 * log10(sum(ATR14) / (n * log(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(rolling_sum(ATR14, 14) / (14 * log10(14))) / log10(14)
    # Even simpler approximation: CHOP = 100 * log10(rolling_sum(ATR14, 14) / (highest_high - lowest_low over 14)) / log10(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, sum_atr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    # For breakout strategy, we want trending regime (CHOP < 38.2)
    trending_regime = chop_aligned < 38.2
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from previous 4h bar (R3, S3)
    prev_close = np.concatenate([[np.nan], close_4h[:-1]])
    prev_high = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low = np.concatenate([[np.nan], low_4h[:-1]])
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 34, 14)  # EMA34, vol MA, ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_val = ema_34_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        chop_val = chop_aligned[i]
        is_trending = trending_regime[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above R3 with uptrend (close > EMA34) and volume spike and trending regime
            long_signal = (high_val > r3_val) and (close_val > ema_34_val) and volume_spike and is_trending
            # Short: price breaks below S3 with downtrend (close < EMA34) and volume spike and trending regime
            short_signal = (low_val < s3_val) and (close_val < ema_34_val) and volume_spike and is_trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S3 (exit long)
            if close_val < s3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Regime change to ranging (exit if chop > 61.8)
            elif chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R3 (exit short)
            if close_val > r3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Regime change to ranging (exit if chop > 61.8)
            elif chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0