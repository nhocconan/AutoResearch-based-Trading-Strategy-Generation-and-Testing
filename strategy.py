#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with ADX regime filter.
In bull regime (ADX > 25): go long when Bull Power > 0 and rising (2-bar momentum).
In bear regime (ADX > 25): go short when Bear Power > 0 and rising (2-bar momentum).
In range regime (ADX < 20): fade extremes using RSI(14) < 30 long, > 70 short.
Uses 6h primary timeframe with 1d HTF for ADX/EMA13 context. Designed for low trade frequency (~15-30/year) to minimize fee drag and work in all market conditions.
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d EMA13 for Elder Ray calculation ===
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # === 1d ADX (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and ADX
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # === Primary timeframe (6h) indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # RSI(14) for range regime
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(adx_14_aligned[i]) or
            np.isnan(rsi_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_high = high[i]
        price_low = low[i]
        price_close = close[i]
        ema13 = ema_13_1d_aligned[i]
        adx = adx_14_aligned[i]
        rsi = rsi_14[i]
        
        # Elder Ray components
        bull_power = price_high - ema13
        bear_power = ema13 - price_low
        
        # Momentum (2-bar change)
        if i >= 2:
            bull_power_mom = bull_power - (prices['high'].iloc[i-2] - ema_13_1d_aligned[i-2])
            bear_power_mom = bear_power - (ema_13_1d_aligned[i-2] - prices['low'].iloc[i-2])
        else:
            bull_power_mom = 0
            bear_power_mom = 0
        
        if position == 0:
            # Regime-based entries
            if adx > 25:  # Trending regime
                if bull_power > 0 and bull_power_mom > 0:  # Strong bullish momentum
                    signals[i] = 0.25
                    position = 1
                elif bear_power > 0 and bear_power_mom > 0:  # Strong bearish momentum
                    signals[i] = -0.25
                    position = -1
            elif adx < 20:  # Range regime
                if rsi < 30:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi > 70:  # Overbought
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:  # Long exit
                if adx > 25 and bear_power > 0 and bear_power_mom > 0:  # Bearish momentum in trend
                    signals[i] = 0.0
                    position = 0
                elif adx < 20 and rsi > 50:  # Range exit at midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short exit
                if adx > 25 and bull_power > 0 and bull_power_mom > 0:  # Bullish momentum in trend
                    signals[i] = 0.0
                    position = 0
                elif adx < 20 and rsi < 50:  # Range exit at midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_v1"
timeframe = "6h"
leverage = 1.0