#!/usr/bin/env python3
"""
6h_WeeklyPivot_HTFBreakout_LTFVolume_Confirmation_v1
Hypothesis: 6h breakouts of weekly pivot levels (R2/S2) with 1d EMA50 trend filter and volume confirmation (1.8x 24-period average) capture institutional interest at key weekly levels. Works in both bull/bear regimes by aligning with HTF trend. Uses discrete sizing (0.25) to minimize fee drag. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA, 1w for pivot)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Weekly pivot points (R2, S2) based on previous week's OHLC ===
    # Using previous week's data for pivot calculation (standard practice)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R2: P + 2*(H - L)
    r2_1w = pivot_1w + 2.0 * (high_1w - low_1w)
    # Weekly S2: P - 2*(H - L)
    s2_1w = pivot_1w - 2.0 * (high_1w - low_1w)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 1.8x 24-period average) ===
    volume = prices['volume'].values
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        pivot_val = pivot_1w_aligned[i]
        r2_val = r2_1w_aligned[i]
        s2_val = s2_1w_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > ema_50_1d_val
        is_bear = price < ema_50_1d_val
        
        if position == 0:
            if is_bull:
                # Bull regime: favor long breakouts above R2
                long_condition = (price > r2_val) and vol_conf
                # Only allow shorts if strong breakdown below S2 with volume
                short_condition = (price < s2_val) and vol_conf and (price < ema_50_1d_val * 0.98)
            else:  # bear regime
                # Bear regime: favor short breakdowns below S2
                short_condition = (price < s2_val) and vol_conf
                # Only allow longs if strong breakout above R2 with volume
                long_condition = (price > r2_val) and vol_conf and (price > ema_50_1d_val * 1.02)
            
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
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR) - wider for 6h timeframe
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below weekly pivot (failed breakout)
                elif price < pivot_val:
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
                # Exit if price breaks above weekly pivot (failed breakdown)
                elif price > pivot_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_HTFBreakout_LTFVolume_Confirmation_v1"
timeframe = "6h"
leverage = 1.0