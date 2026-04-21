#!/usr/bin/env python3
"""
6h_WilliamsAlligator_ElderRay_Regime_V1
Hypothesis: 6h strategy combining Williams Alligator (trend detection) with Elder Ray Index (bull/bear power) filtered by ADX regime.
- Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs on median price. Trend when aligned (Lips > Teeth > Jaw for uptrend).
- Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low. Measures buying/selling pressure.
- ADX > 25 = trending regime (use Alligator/Elder Ray), ADX < 20 = ranging regime (fade extremes).
- Enter long when: Alligator bullish alignment AND Bull Power > 0 AND ADX > 25.
- Enter short when: Alligator bearish alignment AND Bear Power > 0 AND ADX > 25.
- Exit on opposite signal or ATR(10) trailing stop (2.0*ATR).
- Uses 1d HTF for trend context (price > 1d EMA50 for long bias, < for short bias).
Target: 12-25 trades/year (~50-100 total over 4 years) to minimize fee drag.
Works in bull/bear via ADX regime filter and HTF trend bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend bias)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for HTF trend bias ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Williams Alligator: SMAs of median price
    median_price = (high_6h + low_6h) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period
    
    # Elder Ray Index: Bull/Bear Power vs EMA(13)
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema_13
    bear_power = ema_13 - low_6h
    
    # ADX (14-period) for regime detection
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    up_move = pd.Series(high_6h).diff()
    down_move = pd.Series(low_6h).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Handle division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = np.where(np.isnan(adx), 0, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) 
            or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) 
            or np.isnan(adx[i]) or np.isnan(atr[i]) 
            or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Alligator alignment
            alligator_bullish = lips[i] > teeth[i] > jaw[i]
            alligator_bearish = lips[i] < teeth[i] < jaw[i]
            
            # Elder Ray power
            bull_strong = bull_power[i] > 0
            bear_strong = bear_power[i] > 0
            
            # Regime filters
            trending_regime = adx[i] > 25
            ranging_regime = adx[i] < 20
            htf_bullish = price > ema_50_1d_aligned[i]
            htf_bearish = price < ema_50_1d_aligned[i]
            
            # Entry logic
            if alligator_bullish and bull_strong and trending_regime and htf_bullish:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif alligator_bearish and bear_strong and trending_regime and htf_bearish:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Alligator turns bearish OR Bear Power becomes strong
            elif lips[i] < teeth[i] or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Alligator turns bullish OR Bull Power becomes strong
            elif lips[i] > teeth[i] or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_Regime_V1"
timeframe = "6h"
leverage = 1.0