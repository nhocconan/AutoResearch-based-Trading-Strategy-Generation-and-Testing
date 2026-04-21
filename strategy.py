#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_VolumeSpike_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX>25 for trending, ADX<20 for ranging) and volume confirmation (>1.5x 20-period MA). In trending markets (ADX>25), take trend-following entries: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling. In ranging markets (ADX<20), take mean-reversion entries: long when Bull Power crosses above zero from below, short when Bear Power crosses below zero from above. Designed to work in both bull and bear markets by adapting to regime. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for regime and EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d EMA13 for Elder Ray calculation ===
    close_1d = df_1d['close'].values
    ema_13_1d = calculate_ema(close_1d, 13)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # === 1d ADX (14-period) for regime detection ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d_adx, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d_adx, 1)))
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
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h EMA13 for Elder Ray calculation ===
    close = prices['close'].values
    ema_13_6h = calculate_ema(close, 13)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.5x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Elder Ray: Bull Power and Bear Power ===
    bull_power = close - ema_13_6h  # Bull Power = Close - EMA13
    bear_power = ema_13_6h - close  # Bear Power = EMA13 - Close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    prev_bull_power = 0.0
    prev_bear_power = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13_6h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_13_6h_val = ema_13_6h[i]
        vol_avg = vol_ma[i]
        adx_val = adx_1d_aligned[i]
        ema_13_1d_val = ema_13_1d_aligned[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume_now > 1.5 * vol_avg
        
        # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            if is_trending and volume_confirm:
                # Trending market: trend-following entries
                # Long: Bull Power > 0 and rising (bullish momentum)
                long_condition = (bull_power_val > 0) and (bull_power_val > prev_bull_power)
                # Short: Bear Power > 0 and rising (note: Bear Power is positive when close < EMA13)
                short_condition = (bear_power_val > 0) and (bear_power_val > prev_bear_power)
            elif is_ranging and volume_confirm:
                # Ranging market: mean-reversion entries
                # Long: Bull Power crosses above zero from below
                long_condition = (bull_power_val > 0) and (prev_bull_power <= 0)
                # Short: Bear Power crosses above zero from below (Bear Power rising through zero)
                short_condition = (bear_power_val > 0) and (prev_bear_power <= 0)
            else:
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (for trending) or mean reversion exit (for ranging)
                elif is_trending and (bull_power_val <= 0 or bull_power_val < prev_bull_power):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                elif is_ranging and bull_power_val < 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (for trending) or mean reversion exit (for ranging)
                elif is_trending and (bear_power_val <= 0 or bear_power_val < prev_bear_power):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                elif is_ranging and bear_power_val < 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
        
        # Store previous values for next iteration
        prev_bull_power = bull_power_val
        prev_bear_power = bear_power_val
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0