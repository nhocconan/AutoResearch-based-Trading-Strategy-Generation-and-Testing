#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_WeeklyTrend_v1
Hypothesis: 6h Camarilla R1/S1 breakouts with 1d trend regime and weekly pivot direction filter.
Uses discrete sizing (0.25), volume confirmation (1.8x), and ATR stoploss (2.0x).
Target: 50-150 total trades over 4 years for BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for trend regime and pivots
    df_1w = get_htf_data(prices, '1w')  # for weekly pivot direction
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d HMA21 for trend regime ===
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # === 1d ATR (14-period) for stoploss ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d_arr, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d_arr, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d volume confirmation (volume > 1.8x 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > (1.8 * vol_ma_20_1d)
    volume_confirmed_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d)
    
    # === 6h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    # === Weekly pivot direction (from previous week) ===
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = weekly_pivot + (weekly_high - weekly_low) * 1.1 / 12.0
    weekly_s1 = weekly_pivot - (weekly_high - weekly_low) * 1.1 / 12.0
    
    # Align weekly data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly trend: price above/below weekly pivot
    weekly_bull = weekly_pivot_aligned > 0  # valid aligned value
    weekly_bear = weekly_pivot_aligned > 0  # will be replaced with actual comparison
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed_1d_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        hma_21_1d_val = hma_21_1d_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed_1d_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        
        # Weekly trend regime
        weekly_bull = price > weekly_pivot_val
        weekly_bear = price < weekly_pivot_val
        
        if position == 0:
            if weekly_bull:
                # Weekly bull: long breakouts favored
                long_condition = (price > r1_val) and vol_conf
                short_condition = (price < s1_val) and vol_conf and (price < hma_21_1d_val * 0.995)
            else:  # weekly bear
                # Weekly bear: short breakdowns favored
                short_condition = (price < s1_val) and vol_conf
                long_condition = (price > r1_val) and vol_conf and (price > hma_21_1d_val * 1.005)
            
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
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below S1 (failed breakout)
                elif price < s1_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks above R1 (failed breakdown)
                elif price > r1_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(series).ewm(span=half, adjust=False).mean()
    wma1 = pd.Series(series).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma2 - wma1
    hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False).mean()
    return hma.values

name = "6h_Camarilla_R1_S1_Breakout_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0