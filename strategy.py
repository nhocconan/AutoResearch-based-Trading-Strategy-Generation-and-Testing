#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_VolumeFilter_v1
Hypothesis: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with regime filter (ADX>25 for trending, ADX<20 for range) and volume confirmation (>1.5x 20-period average). In trending regimes (ADX>25), trade in direction of Elder Ray power (long if Bull Power>0 & rising, short if Bear Power>0 & rising). In range regimes (ADX<20), fade extreme Elder Ray values (long if Bear Power<-0.5*ATR, short if Bull Power>0.5*ATR). Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-30 trades/year per symbol for low fee drag and strong test generalization across bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for ADX regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d OHLC for ADX calculation (regime filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h EMA13 for Elder Ray calculation ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # === ATR (14-period) for stoploss and regime thresholds ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume filter: 20-period average ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13[i]) or np.isnan(atr[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx = adx_1d_aligned[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        bull = bull_power[i]
        bear = bear_power[i]
        atr_val = atr[i]
        
        if position == 0:
            # Volume filter: current volume > 1.5x 20-period average
            vol_filter = vol_current > 1.5 * vol_average
            
            if adx > 25:  # Trending regime
                # Long conditions: Bull Power > 0 and rising (current > previous)
                bull_rising = bull > bull_power[i-1] if i > 0 else False
                long_signal = bull > 0 and bull_rising and vol_filter
                
                # Short conditions: Bear Power > 0 and rising (current > previous)
                bear_rising = bear > bear_power[i-1] if i > 0 else False
                short_signal = bear > 0 and bear_rising and vol_filter
                
            elif adx < 20:  # Range regime
                # Long conditions: Bear Power < -0.5 * ATR (extreme bearish)
                long_signal = bear < -0.5 * atr_val and vol_filter
                
                # Short conditions: Bull Power > 0.5 * ATR (extreme bullish)
                short_signal = bull > 0.5 * atr_val and vol_filter
                
            else:  # Transition regime (20 <= ADX <= 25) - no trading
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit conditions: reverse signal or regime change
            elif (adx > 25 and bull <= 0) or (adx < 20 and bear >= -0.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit conditions: reverse signal or regime change
            elif (adx > 25 and bear <= 0) or (adx < 20 and bull <= 0.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0