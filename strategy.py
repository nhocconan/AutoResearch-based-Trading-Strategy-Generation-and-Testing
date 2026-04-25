#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R1/S1 breakout on 4h timeframe with daily EMA50 trend filter, volume spike confirmation (>2.0x average), and choppiness regime filter (CHOP < 40) to avoid whipsaw. Designed to capture strong momentum moves in both bull and bear markets by requiring alignment with daily trend and volume expansion. Uses discrete sizing (0.25) and ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR or trend/chop regime breaks). Targets 20-40 trades/year on 4h timeframe to minimize fee drag while capturing high-probability breakouts.
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
    
    # Get 1d data for Camarilla levels, EMA trend, ATR, and choppiness - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for R1 and S1 on 1d
    camarilla_range = high_1d - low_1d
    r1_1d = close_1d + camarilla_range * 1.1 / 12
    s1_1d = close_1d - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate EMA(50) on 1d for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate ATR(14) for stoploss on 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Choppiness Index on 1d for regime filter
    tr_1d1 = high_1d[1:] - low_1d[1:]
    tr_1d2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr_1d3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d_chop = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    atr_sum = pd.Series(atr_1d_chop).rolling(window=14, min_periods=14).sum().values
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    denominator = np.log10(14) * (hh_1d - ll_1d)
    chop_raw = 100 * np.log10(atr_sum / denominator) / np.log10(14)
    chop_raw = np.where((denominator <= 0) | np.isnan(atr_sum) | np.isnan(denominator), 50.0, chop_raw)
    chop_1d = chop_raw
    
    # Align Choppiness Index to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume average (20-period) for volume spike filter on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14, 14)  # EMA needs 50, vol needs 20, ATR needs 14, CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
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
        s1_val = s1_aligned[i]
        ema_val = ema_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        chop_val = chop_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # Choppiness regime filter: only trade when market is strongly trending (CHOP < 40)
        trending_regime = chop_val < 40
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend, volume, and regime confirmation
            # Long: price breaks above R1, above daily EMA50, with volume spike, in trending regime
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike and trending_regime
            # Short: price breaks below S1, below daily EMA50, with volume spike, in trending regime
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
            # 2. Trend reversal: price closes below daily EMA50
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
            # 2. Trend reversal: price closes above daily EMA50
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0