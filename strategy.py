#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter, volume spike, and chop regime filter. 
Targets 75-200 total trades over 4 years by requiring confluence of 1d trend, volume spike, low chop (trending market), and price touching Camarilla levels. 
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via 1d trend filter and chop regime. 
Primary timeframe: 4h, HTF: 1d for trend and Camarilla levels.
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
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla pivot levels (R1, S1) from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h (no extra delay needed as they're based on completed daily candle)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: volume > 2.0x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Choppiness regime filter: CHOP < 50 indicates trending market (favor breakouts)
    # CHOP = 100 * log10(sum(ATR over period) / (log10(highest_high - lowest_low) * period))
    # Simplified: use rolling max/min and ATR
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_series = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_series.sum() / (np.log10(highest_high - lowest_low) * 14)) if (highest_high - lowest_low).iloc[-1] > 0 else 100
    # Vectorized chop calculation
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero or log of non-positive
    hh_ll = hh - ll
    chop = np.full_like(close, 50.0, dtype=float)  # default to neutral
    mask = (hh_ll > 0) & ~np.isnan(hh_ll) & ~np.isnan(atr)
    chop[mask] = 100 * np.log10(atr[mask] / (np.log10(hh_ll[mask]) * 14))
    # Regime: chop < 50 = trending (good for breakouts), chop > 50 = ranging (avoid breakouts)
    chop_regime = chop < 50
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 2 for 1d Camarilla, 20 for volume median, 14 for chop
    start_idx = max(34, 2, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop_regime[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price crosses above Camarilla R1 with volume spike, uptrend (close > EMA34_1d), and trending regime (chop < 50)
            long_entry = (close_val > camarilla_r1_aligned[i]) and vol_spike and (close_val > ema_34_val) and chop_val
            # Short: price crosses below Camarilla S1 with volume spike, downtrend (close < EMA34_1d), and trending regime (chop < 50)
            short_entry = (close_val < camarilla_s1_aligned[i]) and vol_spike and (close_val < ema_34_val) and chop_val
            
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
            # Long - exit on trend reversal or price crosses below Camarilla S1 (mean reversion)
            if close_val < ema_34_val or close_val < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price crosses above Camarilla R1 (mean reversion)
            if close_val > ema_34_val or close_val > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0