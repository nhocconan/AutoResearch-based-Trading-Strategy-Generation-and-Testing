#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_RegimeFilter
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter, volume confirmation, and choppiness regime filter.
Long when price breaks above Camarilla R3 AND price > 12h EMA50 AND volume spike AND chop > 61.8 (range).
Short when price breaks below Camarilla S3 AND price < 12h EMA50 AND volume spike AND chop > 61.8 (range).
Exit on opposite Camarilla level (R3/S3) break or loss of 12h EMA50 alignment.
Designed for 20-50 trades/year on 4h to minimize fee drag while capturing strong intraday moves aligned with 12h trend.
The chop filter (CHOP > 61.8) ensures we only trade in ranging markets, avoiding whipsaws in strong trends.
Works in bull markets (breakouts with 12h uptrend) and bear markets (breakdowns with 12h downtrend) but only during ranging regimes.
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
    
    # Choppiness Index (CHOP) - higher values indicate ranging market
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    # We'll use a simplified version: CHOP = 100 * log10(ATR_sum / (rolling_max - rolling_min)) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(pd.Series(atr).rolling(window=atr_period, min_periods=atr_period).sum().values / range_hl) / np.log10(atr_period)
    chop_filter = chop > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1d Camarilla (2d), 12h EMA50 (~60 4h bars), volume avg, chop
    start_idx = max(48, 60, 20, atr_period*2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        chop_regime = chop_filter[i]
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with 12h EMA50 alignment, volume spike, and ranging regime
            # Long: Close > Camarilla R3 AND price > 12h EMA50 AND volume spike AND chop > 61.8 (ranging)
            # Short: Close < Camarilla S3 AND price < 12h EMA50 AND volume spike AND chop > 61.8 (ranging)
            long_condition = (close_val > r3_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            chop_regime)
            short_condition = (close_val < s3_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             chop_regime)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Camarilla S3 OR loses 12h EMA50 alignment
            if close_val < s3_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R3 OR loses 12h EMA50 alignment
            if close_val > r3_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0