#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_VolumeConfirm_v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1d regime (ADX>25 = trend, ADX<20 = range) and volume confirmation (>1.5x average) captures strong directional moves while avoiding whipsaws. In trend regime (ADX>25): go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising. In range regime (ADX<20): fade extremes (long when Bull Power < -0.5*ATR, short when Bear Power < -0.5*ATR). Designed for 6h to target 12-37 trades/year with discrete sizing (0.25). Works in bull/bear via regime adaptation.
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
    
    # Load 1d data ONCE before loop for Elder Ray and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d High, Low for Bull/Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema_13_1d
    # Bear Power = EMA13 - Low
    bear_power = ema_13_1d - low_1d
    
    # 1d ATR(14) for regime and thresholds
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1)) if len(df_1d) > 1 else np.zeros_like(high_1d)
    tr3 = np.abs(low_1d - np.roll(close_1d, 1)) if len(df_1d) > 1 else np.zeros_like(high_1d)
    if len(df_1d) > 1:
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
    else:
        tr = tr1
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d ADX(14) for regime detection
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 6h (wait for completed 1d bar)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h ATR(14) for stoploss
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # 6h average volume for confirmation (24-period SMA = 2d * 4 = 8d? Actually 6h: 4 bars/day, so 24 = 6d)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of EMA(13), ATR(14), ADX(14), volume(24)
    start_idx = max(13, 14, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_13_1d_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        atr_6h_val = atr_6h[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(bull_val) or np.isnan(bear_val) or 
            np.isnan(atr_1d_val) or np.isnan(adx_val) or np.isnan(avg_vol) or 
            np.isnan(atr_6h_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Regime detection
        trending = adx_val > 25
        ranging = adx_val < 20
        
        # Initialize signal as hold
        signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
        
        if trending and volume_confirmed:
            # Trend regime: follow Elder Ray momentum
            # Long when Bull Power > 0 and rising (bullish momentum)
            # Short when Bear Power > 0 and rising (bearish momentum)
            if i > start_idx:
                bull_prev = bull_power_aligned[i-1]
                bear_prev = bear_power_aligned[i-1]
                bull_rising = bull_val > bull_prev
                bear_rising = bear_val > bear_prev
                
                long_condition = (bull_val > 0) and bull_rising
                short_condition = (bear_val > 0) and bear_rising
                
                if long_condition and position != 1:
                    signals[i] = base_size
                    position = 1
                    entry_price = close_val
                elif short_condition and position != -1:
                    signals[i] = -base_size
                    position = -1
                    entry_price = close_val
        elif ranging and volume_confirmed:
            # Range regime: fade Elder Ray extremes
            # Long when Bull Power is very negative (oversold)
            # Short when Bear Power is very negative (overbought)
            long_condition = bull_val < (-0.5 * atr_1d_val)
            short_condition = bear_val < (-0.5 * atr_1d_val)
            
            if long_condition and position != 1:
                signals[i] = base_size
                position = 1
                entry_price = close_val
            elif short_condition and position != -1:
                signals[i] = -base_size
                position = -1
                entry_price = close_val
        
        # Stoploss: ATR-based (2.0 * ATR)
        if position == 1 and close_val < entry_price - 2.0 * atr_6h_val:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close_val > entry_price + 2.0 * atr_6h_val:
            signals[i] = 0.0
            position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0