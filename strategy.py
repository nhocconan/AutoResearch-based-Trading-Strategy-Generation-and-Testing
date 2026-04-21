#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_WeeklyPivotFilter_v1
Hypothesis: On 6h timeframe, Camarilla R1/S1 breakouts with daily EMA34 trend filter and weekly pivot regime filter capture institutional breakouts in both bull and bear markets. 
In bull regime (price > daily EMA34 AND price > weekly pivot), favor longs on R1 breakouts; in bear regime (price < daily EMA34 AND price < weekly pivot), favor shorts on S1 breakouts. 
Weekly pivot acts as a higher-timeframe structural filter to avoid counter-trend trades. Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for daily trend, 1w for weekly pivot)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Weekly pivot points (based on prior week OHLC) ===
    # Weekly pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    # === 6h Camarilla levels (based on previous bar's OHLC) ===
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_open = np.roll(prices['open'].values, 1)
    prev_close[0] = prev_high[0] = prev_low[0] = prev_open[0] = np.nan  # first bar invalid
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 4 days (8 * 6h = 96h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime + Weekly pivot regime
        is_bull = price > ema_34_1d_val and price > weekly_pivot_val
        is_bear = price < ema_34_1d_val and price < weekly_pivot_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long R1 breakouts favored
                long_condition = (price > r1_val) and vol_conf
                short_condition = (price < s1_val) and vol_conf and (price < ema_34_1d_val * 0.99)  # stricter for shorts
            else:  # bear regime
                # Bear regime: short S1 breakouts favored
                short_condition = (price < s1_val) and vol_conf
                long_condition = (price > r1_val) and vol_conf and (price > ema_34_1d_val * 1.01)  # stricter for longs
            
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
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
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
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_WeeklyPivotFilter_v1"
timeframe = "6h"
leverage = 1.0