#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_RegimeFilter
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter, volume confirmation, and choppiness regime filter.
Long when price breaks above Camarilla R3 AND price > 12h EMA50 AND volume spike AND chop < 61.8 (trending).
Short when price breaks below Camarilla S3 AND price < 12h EMA50 AND volume spike AND chop < 61.8.
Exit on opposite Camarilla level break or loss of 12h EMA50 alignment or chop > 61.8 (choppy regime).
Designed for 15-30 trades/year on 4h to minimize fee drag while capturing strong intraday moves aligned with 12h trend.
Works in bull markets (breakouts with 12h uptrend) and bear markets (breakdowns with 12h downtrend) by avoiding choppy regimes.
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
    
    # Calculate Camarilla levels from prior day (1d)
    df_1d = get_htf_data(prices, '1d')
    # Prior day OHLC (shifted by 1 to avoid look-ahead)
    prev_close = pd.Series(df_1d['close'].values).shift(1)
    prev_high = pd.Series(df_1d['high'].values).shift(1)
    prev_low = pd.Series(df_1d['low'].values).shift(1)
    
    # Camarilla levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR(1) over 14) / (log10(highest high - lowest low over 14)))
    # We'll use a simplified version: CHOP = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Where CHOP > 61.8 = ranging/choppy, CHOP < 38.2 = trending
    tr1 = np.maximum(high[1:] - low[:-1], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop = chop_raw  # already in 0-100 range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1d Camarilla (2d), 12h EMA50 (~60 4h bars), volume avg, chop
    start_idx = max(48, 60, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        in_trending_regime = chop_val < 61.8
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with 12h EMA50 alignment, volume spike, and trending regime
            # Long: Close > Camarilla R3 AND price > 12h EMA50 AND volume spike AND trending regime
            # Short: Close < Camarilla S3 AND price < 12h EMA50 AND volume spike AND trending regime
            long_condition = (close_val > r3_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            in_trending_regime)
            short_condition = (close_val < s3_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             in_trending_regime)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Camarilla S3 OR loses 12h EMA50 alignment OR enters choppy regime
            if close_val < s3_val or close_val < ema_val or not in_trending_regime:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R3 OR loses 12h EMA50 alignment OR enters choppy regime
            if close_val > r3_val or close_val > ema_val or not in_trending_regime:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0