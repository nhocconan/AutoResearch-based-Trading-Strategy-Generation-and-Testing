#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_Filter_v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) on 6h with 1d regime filter (ADX > 25 = trend, ADX < 20 = range). In trend: follow Elder Ray divergence (Bull Power rising while price falling = long; Bear Power rising while price rising = short). In range: mean revert at extremes (Bull Power < -0.5*ATR = long, Bear Power < -0.5*ATR = short). Uses discrete sizing 0.25 to minimize fee drag. Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for HTF regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # 6h ATR for dynamic thresholds
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low_1d[1:] - close[:-1]) if len(low) > 1 else np.array([])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))]) if len(low) > 1 else np.full_like(close, np.nan)
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(30, 13, 14)  # 1d ADX, 6h EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(atr_6h[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        adx_val = adx_aligned[i]
        ema_13_val = ema_13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr_6h[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Regime detection: ADX > 25 = trend, ADX < 20 = range, 20-25 = transition (hold)
        is_trend = adx_val > 25
        is_range = adx_val < 20
        
        if position == 0:
            # Look for entry signals
            long_signal = False
            short_signal = False
            
            if is_trend:
                # Trend regime: follow Elder Ray divergence
                # Bullish divergence: Bull Power rising (current > previous) while price weak
                # Bearish divergence: Bear Power rising (current > previous) while price strong
                if i > 0:
                    bull_rising = bull_val > bull_power[i-1]
                    bear_rising = bear_val > bear_power[i-1]
                    price_weak = close_val < close[i-1]  # Price falling
                    price_strong = close_val > close[i-1]  # Price rising
                    
                    long_signal = bull_rising and price_weak and (close_val > ema_13_val)
                    short_signal = bear_rising and price_strong and (close_val < ema_13_val)
            
            elif is_range:
                # Range regime: mean revert at extremes
                # Long when Bull Power is very negative (selling exhaustion)
                # Short when Bear Power is very negative (buying exhaustion)
                long_signal = bull_val < -0.5 * atr_val
                short_signal = bear_val < -0.5 * atr_val
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if is_trend:
                # Exit on bearish divergence or price below EMA13
                if i > 0:
                    bear_rising = bear_val > bear_power[i-1]
                    price_strong = close_val > close[i-1]
                    if bear_rising and price_strong:
                        signals[i] = 0.0
                        position = 0
            else:  # range regime
                # Exit when Bull Power returns to neutral
                if bull_val > -0.2 * atr_val:
                    signals[i] = 0.0
                    position = 0
        
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if is_trend:
                # Exit on bullish divergence or price above EMA13
                if i > 0:
                    bull_rising = bull_val > bull_power[i-1]
                    price_weak = close_val < close[i-1]
                    if bull_rising and price_weak:
                        signals[i] = 0.0
                        position = 0
            else:  # range regime
                # Exit when Bear Power returns to neutral
                if bear_val > -0.2 * atr_val:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_Filter_v1"
timeframe = "6h"
leverage = 1.0