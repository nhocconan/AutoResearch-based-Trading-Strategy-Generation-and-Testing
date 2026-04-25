#!/usr/bin/env python3
"""
6h Elder Ray + Regime Filter (ADX + Chop)
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure.
Combine with ADX trend strength and Chop regime filter to avoid whipsaws. In strong trends (ADX>25), 
take Elder Ray signals in trend direction. In chop (Chop>61.8), fade extreme Elder Ray readings.
Uses 6h timeframe for lower frequency and 1d/1w HTF for regime context. Targets 12-37 trades/year.
Works in bull/bear by adapting to regime.
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
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for super HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Calculate 1d Bull Power (High - EMA13) and Bear Power (EMA13 - Low)
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = ema_13_1d - df_1d['low'].values
    
    # Align Elder Ray components to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d ADX for trend strength
    # True Range
    tr1 = pd.Series(df_1d['high']).rolling(2).max() - pd.Series(df_1d['low']).rolling(2).min()
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = -pd.Series(df_1d['low']).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    # Smoothed TR, DM+
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / np.maximum(tr_14, 1e-10)
    di_minus = 100 * dm_minus_14 / np.maximum(tr_14, 1e-10)
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1w EMA50 for super trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Chop filter on 6h to detect regime
    tr1_6h = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2_6h = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3_6h = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    sum_atr_6h = pd.Series(atr_6h).rolling(window=14, min_periods=14).sum().values
    hh_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_6h = 100 * np.log10(sum_atr_6h / np.maximum(hh_6h - ll_6h, 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(13, 14, 14, 50, 14)  # EMA13, ADX, ATR, EMA50, Chop
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(chop_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        adx = adx_aligned[i]
        ema_50_1w = ema_50_1w_aligned[i]
        chop = chop_6h[i]
        
        # Determine regime
        strong_trend = adx > 25
        ranging_market = chop > 61.8
        super_uptrend = curr_close > ema_50_1w
        super_downtrend = curr_close < ema_50_1w
        
        if position == 0:
            # Look for entry signals
            long_entry = False
            short_entry = False
            
            if strong_trend:
                # In strong trend: follow Elder Ray momentum in trend direction
                if super_uptrend and bull_power > 0:
                    long_entry = True
                elif super_downtrend and bear_power > 0:
                    short_entry = True
            elif ranging_market:
                # In chop: fade extreme Elder Ray (mean reversion)
                if bull_power < -np.std(bull_power_aligned[max(0, i-100):i+1]):  # extreme selling pressure
                    long_entry = True
                elif bear_power < -np.std(bear_power_aligned[max(0, i-100):i+1]):  # extreme buying pressure
                    short_entry = True
            else:
                # Weak trend: moderate Elder Ray filter
                if super_uptrend and bull_power > 0:
                    long_entry = True
                elif super_downtrend and bear_power > 0:
                    short_entry = True
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Elder Ray turns negative OR loss of super trend OR chop becomes extreme
            if (bull_power <= 0) or (not super_uptrend) or (chop > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bear Power turns negative OR loss of super trend OR chop becomes extreme
            if (bear_power <= 0) or (not super_downtrend) or (chop > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Regime_ADX_ChopFilter"
timeframe = "6h"
leverage = 1.0