#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with regime filter (ADX < 20 for mean reversion, ADX > 25 for trend) on 6h timeframe. Uses 1d HTF for EMA context and volume confirmation. Designed to capture both trending and mean-reverting markets with low trade frequency (~15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume ratio (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 6h indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = EMA13 - Low
    bear_power = ema_13 - low
    
    # ADX(14) for regime filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        ema_13_val = ema_13[i]
        bull = bull_power[i]
        bear = bear_power[i]
        adx_val = adx[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        
        if position == 0:
            # Regime: ADX < 20 = range (mean reversion), ADX > 25 = trend
            if adx_val < 20:
                # Mean reversion: fade extremes
                # Long: Bear Power > 0 (bullish momentum) + price below 1d EMA34 + volume confirmation
                if bull > 0 and price_close < trend_1d and vol_spike > 1.5:
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power > 0 (bullish momentum fading) + price above 1d EMA34
                elif bull > 0 and price_close > trend_1d and vol_spike > 1.5:
                    signals[i] = -0.25
                    position = -1
            elif adx_val > 25:
                # Trend following: acceleration
                # Long: Bull Power increasing (bullish acceleration) + price above 1d EMA34
                if i >= 101 and bull > bull_power[i-1] and price_close > trend_1d and vol_spike > 1.5:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power increasing (bearish acceleration) + price below 1d EMA34
                elif i >= 101 and bear > bear_power[i-1] and price_close < trend_1d and vol_spike > 1.5:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long: Bear Power turns negative OR price crosses below EMA13
                if bear < 0 or price_close < ema_13_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Bull Power turns negative OR price crosses above EMA13
                if bull < 0 or price_close > ema_13_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_v1"
timeframe = "6h"
leverage = 1.0