#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 12h Camarilla R3/S3 breakout in direction of 1d EMA34 trend with volume confirmation and chop regime filter.
Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume spike AND chop < 61.8 (trending).
Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume spike AND chop < 61.8 (trending).
Exit on opposite Camarilla breakout (R3/S3) or loss of trend alignment.
Designed for 12-25 trades/year on 12h to minimize fee drag while capturing strong directional moves aligned with daily trend.
Works in bull markets (breakouts with daily uptrend) and bear markets (breakdowns with daily downtrend).
Chop regime filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d data ONCE before loop for EMA34 and Camarilla
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # 1d EMA34 for trend
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Camarilla levels (R3, S3) from prior day
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2.0
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d.values)
    
    # 12h Donchian-like breakout using Camarilla levels (prior day's levels)
    # We use prior day's Camarilla levels as breakout thresholds for current 12h session
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index regime filter (trending when CHOP < 61.8)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (HHV - LLV))) / log10(n)
    # Simplified: use rolling max/min and ATR
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    hhvl = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    llvl = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop_raw = 100 * np.log10(atr * atr_period / (hhvl - llvl)) / np.log10(atr_period)
    # Handle division by zero and invalid values
    chop = np.where((hhvl - llvl) > 0, chop_raw, 50.0)  # default to 50 when range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for EMA34 (34d = ~680 12h bars), Camarilla (2d), volume avg (20), ATR (14)
    start_idx = max(34*2, 2, 20, 14)  # ~68 bars for 1d EMA34 in 12h
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        in_trending_regime = chop_val < 61.8
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with trend alignment, volume spike, and trending regime
            # Long: Close > Camarilla R3 AND close > 1d EMA34 AND volume spike AND trending regime
            # Short: Close < Camarilla S3 AND close < 1d EMA34 AND volume spike AND trending regime
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
            # Long - exit when price breaks below Camarilla S3 OR loses trend alignment
            if close_val < s3_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R3 OR loses trend alignment
            if close_val > r3_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0