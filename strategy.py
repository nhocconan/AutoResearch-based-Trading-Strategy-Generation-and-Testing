#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike_WeeklyRegime_v1
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation (>1.8x 20-period MA), and weekly ADX regime filter (ADX>25 for trend, ADX<20 for range). 
In bull/bear markets: only trade breakouts aligned with 1d EMA50 trend. In ranging markets (weekly ADX<20): fade at Donchian bands with volume confirmation.
Uses ATR-based stop (2.5x) and minimum holding period of 3 bars to reduce churn.
Designed for 6h timeframe with 1d HTF trend and weekly HTF regime to work in both bull and bear markets by adapting to volatility regimes.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend, 1w for ADX regime)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1w ADX(14) for regime detection ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                                 np.maximum(high_1w - np.roll(high_1w, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                                  np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0))
    
    # Smoothed DM and TR
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = tr_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1w = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w, additional_delay_bars=0)
    
    # === 6h ATR (20-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    # === Volume confirmation (1.8x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 6h Donchian(20) channels ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        adx_1w_val = adx_1w_aligned[i]
        vol_avg = vol_ma[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume confirmation: current volume > 1.8x average (strict threshold)
        volume_confirm = volume_now > 1.8 * vol_avg
        
        # Regime classification
        is_trending = adx_1w_val > 25
        is_ranging = adx_1w_val < 20
        
        if position == 0:
            if is_trending:
                # Trend mode: breakout in direction of 1d EMA50
                long_condition = (price > upper_channel) and (price > ema_50_1d_val) and volume_confirm
                short_condition = (price < lower_channel) and (price < ema_50_1d_val) and volume_confirm
            else:  # ranging mode
                # Range mode: fade at channels with volume confirmation
                long_condition = (price < lower_channel) and volume_confirm  # mean reversion long at support
                short_condition = (price > upper_channel) and volume_confirm  # mean reversion short at resistance
            
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
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below 1d EMA50 in trending mode)
                elif is_trending and price < ema_50_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Range exit: price crosses midpoint
                elif not is_trending and price > (upper_channel + lower_channel) / 2:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above 1d EMA50 in trending mode)
                elif is_trending and price > ema_50_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Range exit: price crosses midpoint
                elif not is_trending and price < (upper_channel + lower_channel) / 2:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike_WeeklyRegime_v1"
timeframe = "6h"
leverage = 1.0