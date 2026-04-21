#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop
Hypothesis: 12h Camarilla pivot R1/S1 breakouts with volume confirmation and choppiness regime filter capture institutional order flow during trend acceleration. HTF 1d trend filter avoids counter-trend trades. Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. Works in bull/bear markets: Camarilla levels adapt to volatility, regime filter prevents whipsaws in ranging markets, volume confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for pivot calculation and trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels (R1, S1, close) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = high_1d - low_1d
    r1_1d = close_1d + (1.1 * camarilla_range / 12)
    s1_1d = close_1d - (1.1 * camarilla_range / 12)
    
    # Align 1d levels to 12h timeframe (wait for 1d candle close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume confirmation: 12h volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * vol_ma)
    
    # Choppiness regime filter: CHOP(14) > 61.8 = ranging (mean revert), < 38.2 = trending
    # We'll use trending regime only for breakouts (CHOP < 38.2)
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean().values
    
    # True range sum for CHOP denominator
    tr_sum = tr.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_14 - ll_14
    chop = np.where(hh_ll_diff > 0, 100 * np.log10(tr_sum / hh_ll_diff) / np.log10(14), 50)
    # Trending regime: CHOP < 38.2
    trending_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Warmup for indicators
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])
            or np.isnan(trending_regime[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        
        if position == 0:
            # Long breakout: price > R1 + volume spike + trending regime + price > EMA34 (long bias)
            if (price > r1_1d_aligned[i] and volume_spike[i] and 
                trending_regime[i] and price > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakout: price < S1 + volume spike + trending regime + price < EMA34 (short bias)
            elif (price < s1_1d_aligned[i] and volume_spike[i] and 
                  trending_regime[i] and price < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss: 2.5 * ATR below entry
            if price < entry_price - 2.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below S1 (failed breakout) or regime changes to ranging
            elif price < s1_1d_aligned[i] or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss: 2.5 * ATR above entry
            if price > entry_price + 2.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above R1 (failed breakout) or regime changes to ranging
            elif price > r1_1d_aligned[i] or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop"
timeframe = "12h"
leverage = 1.0