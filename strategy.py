#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA34 trend filter, volume spike (>2.0x average), and choppiness regime filter (CHOP > 61.8 for mean reversion).
Only trade breakouts aligned with 1d EMA34 direction during high volume expansion in choppy markets.
Designed to capture reversals in ranging markets while avoiding strong trends where breakouts fail.
Uses discrete sizing (0.25) and ATR-based stoploss (2.0x ATR).
Targets 20-50 trades/year on 4h timeframe to minimize fee drag.
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
    
    # Get 4h data for Camarilla levels and ATR - primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate previous period's Camarilla levels (using prior 4h bar's HLC)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for R1, R2, S1, S2
    # R1 = Close + ((High - Low) * 1.1/12)
    # R2 = Close + ((High - Low) * 1.1/6)
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    rang = prev_high - prev_low
    r1 = prev_close + (rang * 1.1 / 12)
    r2 = prev_close + (rang * 1.1 / 6)
    s1 = prev_close - (rang * 1.1 / 12)
    s2 = prev_close - (rang * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    r2_aligned = align_htf_to_ltf(prices, df_4h, r2)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_4h, s2)
    
    # Get 1d data for EMA34 trend filter - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (CHOP) on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high) - min(low))) / log10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_14_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate CHOP(14) on 1d
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop_1d = chop_raw
    
    # Align CHOP to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 20, 14)  # EMA needs 34, vol needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop_1d_aligned[i])):
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
        r2_val = r2_aligned[i]
        s1_val = s1_aligned[i]
        s2_val = s2_aligned[i]
        ema_val = ema_1d_aligned[i]
        atr_val = atr_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        chop_val = chop_1d_aligned[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # Choppiness regime: CHOP > 61.8 indicates ranging market (mean reversion favorable)
        chop_regime = chop_val > 61.8
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend, volume, and chop confirmation
            # Long: price breaks above R1, above 1d EMA34, with volume spike, in choppy market
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike and chop_regime
            # Short: price breaks below S1, below 1d EMA34, with volume spike, in choppy market
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike and chop_regime
            
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
            # 2. Trend reversal: price closes below 1d EMA34
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Regime change: market becomes trending (CHOP < 38.2)
            elif chop_val < 38.2:
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
            # 2. Trend reversal: price closes above 1d EMA34
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Regime change: market becomes trending (CHOP < 38.2)
            elif chop_val < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0