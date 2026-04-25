#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wEMA34_VolumeSpike_ChopRegime
Hypothesis: Camarilla R1/S1 breakout on 12h timeframe with weekly EMA34 trend filter, volume spike confirmation, and choppiness regime filter to avoid whipsaw. 
Only trade breakouts aligned with weekly EMA34 direction during volume expansion (>2.0x average volume) when market is trending (CHOP < 40).
Designed to work in both bull (trend-following breakouts) and bear (mean-reversion at extremes) regimes by filtering for strongly trending conditions only.
Uses discrete sizing (0.25) and ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
Targets 12-37 trades/year on 12h timeframe to minimize fee drag while capturing strong momentum moves.
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
    
    # Get 12h data for Camarilla levels and ATR - primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for R1 and S1
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_range = high_12h - low_12h
    r1_12h = close_12h + camarilla_range * 1.1 / 12
    s1_12h = close_12h - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Get 1w data for EMA34 trend filter - HTF
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate ATR(14) for stoploss on 12h
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 12h for regime filter
    tr_12h1 = high_12h[1:] - low_12h[1:]
    tr_12h2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr_12h3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    
    atr_12h_chop = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate sum of ATR over 14 periods
    atr_sum = pd.Series(atr_12h_chop).rolling(window=14, min_periods=14).sum().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness Index
    denominator = np.log10(14) * (hh_12h - ll_12h)
    chop_raw = 100 * np.log10(atr_sum / denominator) / np.log10(14)
    chop_raw = np.where((denominator <= 0) | np.isnan(atr_sum) | np.isnan(denominator), 50.0, chop_raw)
    chop_12h = chop_raw
    
    # Align Choppiness Index to 12h timeframe
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 20, 14, 14)  # EMA needs 34, vol needs 20, ATR needs 14, CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop_12h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_1w_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        chop_val = chop_12h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average (stricter to reduce trades)
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # Choppiness regime filter: only trade when market is strongly trending (CHOP < 40)
        trending_regime = chop_val < 40
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend, volume, and regime confirmation
            # Long: price breaks above R1, above weekly EMA34, with volume spike, in trending regime
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike and trending_regime
            # Short: price breaks below S1, below weekly EMA34, with volume spike, in trending regime
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike and trending_regime
            
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
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes below weekly EMA34
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Regime change: market becomes choppy (CHOP >= 40)
            elif chop_val >= 40:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes above weekly EMA34
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Regime change: market becomes choppy (CHOP >= 40)
            elif chop_val >= 40:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wEMA34_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0