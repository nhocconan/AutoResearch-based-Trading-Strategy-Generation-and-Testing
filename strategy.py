#!/usr/bin/env python3
"""
12h_PriceAction_Confluence_V1
Hypothesis: Price action confluence at key levels (support/resistance, volume clusters) with trend filter
Works in bull/bear by trading breakouts with trend confirmation and mean reversion in ranges
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
Edge: Combines structure (swing points), volume confirmation, and trend/range adaptation
"""

name = "12h_PriceAction_Confluence_V1"
timeframe = "12h"
leverage = 1.0

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
    
    # Trend filter: ADX(14) on 12h data
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                    else:
                        result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    # Get 12h data for ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Get 1d data for swing points and volume profile
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Swing high/low from previous day (structure)
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    
    # Volume-weighted average price (VWAP) from previous day
    pv = df_1d['volume'].shift(1).values
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pv_tp = typical_price * pv
    vwap_num = pd.Series(pv_tp).shift(1).values
    vwap_den = pd.Series(pv).shift(1).values
    vwap = np.divide(vwap_num, vwap_den, out=np.zeros_like(vwap_num), where=vwap_den!=0)
    
    # Align to 12h
    ph_aligned = align_htf_to_ltf(prices, df_1d, ph)
    pl_aligned = align_htf_to_ltf(prices, df_1d, pl)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Volume confirmation: above average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)  # Moderate threshold
    
    # Volatility filter: ATR ratio to avoid extreme conditions
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.zeros_like(close)
        atr[:period-1] = np.nan
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_12h = calculate_atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    atr_ma = pd.Series(atr_12h_aligned).rolling(window=20, min_periods=20).mean().values
    volatility_normal = (atr_12h_aligned > 0) & (atr_12h_aligned < atr_ma * 3)  # Not too volatile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(ph_aligned[i]) or 
            np.isnan(pl_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime detection: trending vs ranging
        strong_trend = adx_12h_aligned[i] > 25
        ranging_market = adx_12h_aligned[i] < 20
        
        if position == 0:
            # Long conditions
            long_breakout = (close[i] > ph_aligned[i])  # Break above prior day high
            long_pullback = (close[i] > vwap_aligned[i] and close[i] < ph_aligned[i] * 1.02)  # Near VWAP in uptrend
            long_setup = volume_confirm[i] and volatility_normal[i]
            
            if strong_trend and long_breakout and long_setup:
                signals[i] = 0.25
                position = 1
            elif ranging_market and long_pullback and long_setup:
                signals[i] = 0.20
                position = 1
            
            # Short conditions
            short_breakout = (close[i] < pl_aligned[i])  # Break below prior day low
            short_pullback = (close[i] < vwap_aligned[i] and close[i] > pl_aligned[i] * 0.98)  # Near VWAP in downtrend
            short_setup = volume_confirm[i] and volatility_normal[i]
            
            if strong_trend and short_breakout and short_setup:
                signals[i] = -0.25
                position = -1
            elif ranging_market and short_pullback and short_setup:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long exit conditions
            if (close[i] < pl_aligned[i]) or (adx_12h_aligned[i] < 18 and close[i] < vwap_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit conditions
            if (close[i] > ph_aligned[i]) or (adx_12h_aligned[i] < 18 and close[i] > vwap_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals