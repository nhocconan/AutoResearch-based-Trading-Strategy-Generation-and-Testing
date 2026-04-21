#!/usr/bin/env python3
"""
12h_HTF_1w_1d_Camarilla_R4S4_Breakout_Volume_ChopRegime
Hypothesis: 12h Camarilla R4/S4 breakout with weekly EMA50 trend filter and daily choppiness regime filter. 
Only trade when CHOP(14) < 38.2 (trending market) to avoid whipsaws in ranging conditions. 
Volume confirmation (>1.5x 20-period volume MA) reduces false breakouts. 
Target 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.
Uses 12h primary timeframe with 1w/1d HTF for pivot calculation, trend filter, and regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots and chop, 1w for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla pivot points (R4, S4) from previous day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r4 = pivot + (range_ * 1.1 / 2)  # R4 = pivot + range*(1.1/2)
    s4 = pivot - (range_ * 1.1 / 2)  # S4 = pivot - range*(1.1/2)
    
    # Align R4/S4 to 12h timeframe (wait for 1d bar close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First bar has no previous close
    
    # Sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(atr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hl_range = hh_14 - ll_14
    chop = np.where(hl_range > 0, 100 * np.log10(atr_14 / hl_range) / np.log10(14), 100)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        chop_ok = chop_aligned[i] < 38.2  # trending market regime (CHOP < 38.2)
        
        if position == 0:
            # Long: price breaks above R4 + volume confirmation + weekly uptrend + trending regime
            if price > r4_aligned[i] and vol_ok and price > ema_50_1w_aligned[i] and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + volume confirmation + weekly downtrend + trending regime
            elif price < s4_aligned[i] and vol_ok and price < ema_50_1w_aligned[i] and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly EMA50 or regime changes to ranging
            if price < ema_50_1w_aligned[i] or chop_aligned[i] >= 61.8:  # Exit if ranging (CHOP > 61.8)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly EMA50 or regime changes to ranging
            if price > ema_50_1w_aligned[i] or chop_aligned[i] >= 61.8:  # Exit if ranging (CHOP > 61.8)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1w_1d_Camarilla_R4S4_Breakout_Volume_ChopRegime"
timeframe = "12h"
leverage = 1.0