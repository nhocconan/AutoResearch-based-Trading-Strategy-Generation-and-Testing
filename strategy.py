#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_RegimeFilter_ATRStop_v1
Hypothesis: Camarilla pivot R1/S1 breakouts on 4h filtered by 1d choppiness regime (CHOP > 61.8 = range) and volume spike.
In ranging markets (high CHOP), fade breaks of R1/S1 with mean reversion to Pivot point.
In trending markets (low CHOP), breakouts continue in direction of trend.
Uses ATR(14) stoploss (1.5x) and discrete position sizing (0.25) to minimize fee churn.
Designed for 20-40 trades/year per symbol, targeting BTC/ETH robustness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for CHOP and Pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h OHLC for price action ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d OHLC for Camarilla pivot calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # Pivot = (High + Low + Close) / 3
    rng_1d = high_1d - low_1d
    camarilla_r1_1d = close_1d + rng_1d * 1.1 / 12
    camarilla_s1_1d = close_1d - rng_1d * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align 1d Camarilla levels to 4h
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # === 1d Choppiness Index regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (HHV - LLV))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3_1d = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean()
    
    sum_atr_1d = atr_1d.rolling(window=14, min_periods=14).sum()
    hhvl_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max() - \
              pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    
    # Avoid division by zero
    chop_1d = 100 * np.log10(sum_atr_1d / (np.log10(14) * hhvl_1d.replace(0, np.nan))) / np.log10(14)
    chop_1d = chop_1d.fillna(50).values  # neutral when undefined
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === ATR (14-period) for stoploss ===
    tr1_4h = pd.Series(high - low)
    tr2_4h = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3_4h = pd.Series(np.abs(low - np.roll(close, 1)))
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_4h[i]) or np.isnan(camarilla_s1_4h[i]) 
            or np.isnan(pivot_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average (less strict than 2.0x)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_filter = vol > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        if position == 0:
            # Regime-based logic
            if chop_4h[i] > 61.8:  # Ranging market - mean reversion
                # Fade R1/S1 breaks: short at R1, long at S1
                short_at_r1 = price > camarilla_r1_4h[i] and vol_filter
                long_at_s1 = price < camarilla_s1_4h[i] and vol_filter
                
                if short_at_r1:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                elif long_at_s1:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    
            else:  # Trending market - breakout continuation
                # Break R1/S1 in direction of break
                break_r1 = price > camarilla_r1_4h[i] and vol_filter
                break_s1 = price < camarilla_s1_4h[i] and vol_filter
                
                if break_r1:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif break_s1:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Long position management
            # Stoploss: 1.5x ATR below entry
            if price < entry_price - 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Take profit at Pivot level
            elif price >= pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            # Reverse signal: price reaches opposite Camarilla level
            elif price >= camarilla_r1_4h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
                entry_price = price
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position management
            # Stoploss: 1.5x ATR above entry
            if price > entry_price + 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Take profit at Pivot level
            elif price <= pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            # Reverse signal: price reaches opposite Camarilla level
            elif price <= camarilla_s1_4h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
                entry_price = price
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_RegimeFilter_ATRStop_v1"
timeframe = "4h"
leverage = 1.0