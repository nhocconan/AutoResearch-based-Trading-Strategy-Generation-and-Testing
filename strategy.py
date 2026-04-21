#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v2
Hypothesis: Daily Camarilla pivot (R1/S1) breakout filtered by 1w EMA50 trend and volume spike (>1.5x 20-period average).
In trending markets (price > 1w EMA50): breakout continuation (long above R1, short below S1).
In ranging markets (price near 1w EMA50 ± 2%): mean reversion at S1/R1 (long at S1, short at R1).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to balance returns and fee drag.
Designed to work in both bull and bear markets via adaptive regime filter.
Timeframe: 1d, HTF: 1w for trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA50 trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === Daily OHLC for Camarilla pivot calculation (based on previous daily bar) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    
    # Align 1d Camarilla levels to daily timeframe (no shift needed as already aligned)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1w EMA50 for trend filter ===
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Volume spike filter (1d volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Determine market regime based on price vs 1w EMA50
            ema_pct_dev = abs(price - ema_trend) / ema_trend
            is_trending = ema_pct_dev > 0.02  # >2% deviation = trending
            is_ranging = ema_pct_dev <= 0.02   # <=2% deviation = ranging
            
            # Trending market: breakout continuation with volume confirmation
            if is_trending and vol_spike:
                long_condition = price > r1 and price > ema_trend
                short_condition = price < s1 and price < ema_trend
            
            # Ranging market: mean reversion at S1/R1 with volume confirmation
            elif is_ranging and vol_spike:
                long_condition = price <= s1 * 1.002  # slight buffer for entry
                short_condition = price >= r1 * 0.998
            
            # No volume spike or transition zone: no entries
            else:
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime-based exit
            elif (ema_pct_dev <= 0.02 and price >= r1 * 0.998) or (ema_pct_dev > 0.02 and price < s1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime-based exit
            elif (ema_pct_dev <= 0.02 and price <= s1 * 1.002) or (ema_pct_dev > 0.02 and price > r1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v2"
timeframe = "1d"
leverage = 1.0