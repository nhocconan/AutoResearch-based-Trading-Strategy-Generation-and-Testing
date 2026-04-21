#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Regime_v2
Hypothesis: Use daily Camarilla pivot levels (R1/S1) on 4h timeframe with 1d EMA34 trend filter, volume confirmation, and ADX regime filter. 
Long when price breaks above R1 with volume spike and ADX>25 (trending), short when breaks below S1 with volume spike and ADX>25. 
Exit when price returns to pivot point (PP). 
Key improvements: Reduced volume threshold to 1.3 (from 1.5) and added minimum hold period of 3 bars to reduce whipsaw and trade frequency.
Target ~20-50 trades/year on 4h by requiring multiple confluence conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Calculate Camarilla pivot levels (R1, S1, PP) from 1d OHLC ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume confirmation: 20-period volume average on 4h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    # === ADX regime filter (14-period) on 4h ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                bars_since_entry = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_34_1d_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        pp_level = pp_aligned[i]
        vol_spike = vol_ratio[i]
        adx_val = adx[i]
        
        # Regime filter: only trade in trending markets (ADX > 25)
        is_trending = adx_val > 25
        
        if position == 0:
            bars_since_entry = 0
            # Long: Price breaks above R1 + volume spike > 1.3 + above 1d EMA34 + trending
            if (price_close > r1_level and 
                vol_spike > 1.3 and 
                price_close > trend_1d and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume spike > 1.3 + below 1d EMA34 + trending
            elif (price_close < s1_level and 
                  vol_spike > 1.3 and 
                  price_close < trend_1d and 
                  is_trending):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            bars_since_entry += 1
            # Minimum hold period of 3 bars to reduce whipsaw
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                # Exit when price returns to pivot point (PP)
                if position == 1 and price_close < pp_level:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                elif position == -1 and price_close > pp_level:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    # Hold position
                    signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Regime_v2"
timeframe = "4h"
leverage = 1.0