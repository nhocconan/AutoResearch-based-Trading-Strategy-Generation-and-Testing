#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike, and chop regime filter.
Long when price breaks above Camarilla R1 AND price > 1d EMA34 AND volume spike AND chop < 61.8 (trending).
Short when price breaks below Camarilla S1 AND price < 1d EMA34 AND volume spike AND chop < 61.8.
Exit on opposite Camarilla level (S1/R1) break or loss of 1d EMA34 alignment.
Designed for 12-37 trades/year on 12h to minimize fee drag while capturing strong moves aligned with daily trend.
Works in bull markets (breakouts with 1d uptrend) and bear markets (breakdowns with 1d downtrend).
Chop filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior day (1d)
    df_1d = get_htf_data(prices, '1d')
    # Prior day OHLC (shifted by 1 to avoid look-ahead)
    prev_close = pd.Series(df_1d['close'].values).shift(1)
    prev_high = pd.Series(df_1d['high'].values).shift(1)
    prev_low = pd.Series(df_1d['low'].values).shift(1)
    
    # Camarilla levels
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike: current volume > 2.0 * 24-period average (24*12h = 12d)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index (CHOP) regime filter on 12h
    # CHOP(14) > 61.8 = ranging, CHOP < 38.2 = trending
    # We want trending markets: CHOP < 61.8
    atr_period = 14
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # first TR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = 100 * np.log10(atr * np.sqrt(atr_period) / (highest_high - lowest_low)) / np.log10(atr_period)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1d Camarilla (2d), 1d EMA34 (~34 12h bars), volume avg (24), chop (14)
    start_idx = max(48, 34, 24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        in_trending_regime = chop_val < 61.8
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with 1d EMA34 alignment, volume spike, and trending regime
            # Long: Close > Camarilla R1 AND price > 1d EMA34 AND volume spike AND trending regime
            # Short: Close < Camarilla S1 AND price < 1d EMA34 AND volume spike AND trending regime
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            in_trending_regime)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             in_trending_regime)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Camarilla S1 OR loses 1d EMA34 alignment
            if close_val < s1_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R1 OR loses 1d EMA34 alignment
            if close_val > r1_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0