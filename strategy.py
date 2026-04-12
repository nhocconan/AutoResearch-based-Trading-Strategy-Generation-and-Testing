#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Range_Reversion_v1
Hypothesis: Trade reversals at weekly Camarilla pivot levels (H4, L4) on daily timeframe with volume confirmation and RSI extremes. 
In ranging markets (Choppiness Index > 61.8 on weekly), price tends to revert from extreme pivot levels. 
In trending markets (Choppiness Index <= 61.8), breakouts of H4/L4 levels are traded with momentum.
Uses weekly timeframe for regime and pivot calculation, daily for entries. Target: 15-25 trades/year.
Works in bull markets (breakouts continue) and bear markets (reversions from overbought/oversold levels).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Pivot_Range_Reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR PIVOTS AND REGIME ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    # H2 = Close + 0.6 * (High - Low)
    # L2 = Close - 0.6 * (High - Low)
    # H1 = Close + 0.318 * (High - Low)
    # L1 = Close - 0.318 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    
    # Shift by 1 to use previous week's data (no look-ahead)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot levels for previous week
    rng = prev_high - prev_low
    H4 = prev_close + 1.5 * rng
    L4 = prev_close - 1.5 * rng
    H3 = prev_close + 1.1 * rng
    L3 = prev_close - 1.1 * rng
    H2 = prev_close + 0.6 * rng
    L2 = prev_close - 0.6 * rng
    H1 = prev_close + 0.318 * rng
    L1 = prev_close - 0.318 * rng
    Pivot = (prev_high + prev_low + prev_close) / 3
    
    # Weekly Choppiness Index for regime detection (14-period)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr_1w = true_range(high_1w, low_1w, np.roll(close_1w, 1))
    tr_1w[0] = np.nan
    
    atr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # Align weekly data to daily
    H4_d = align_htf_to_ltf(prices, df_1w, H4)
    L4_d = align_htf_to_ltf(prices, df_1w, L4)
    H3_d = align_htf_to_ltf(prices, df_1w, H3)
    L3_d = align_htf_to_ltf(prices, df_1w, L3)
    H2_d = align_htf_to_ltf(prices, df_1w, H2)
    L2_d = align_htf_to_ltf(prices, df_1w, L2)
    H1_d = align_htf_to_ltf(prices, df_1w, H1)
    L1_d = align_htf_to_ltf(prices, df_1w, L1)
    Pivot_d = align_htf_to_ltf(prices, df_1w, Pivot)
    chop_d = align_htf_to_ltf(prices, df_1w, chop)
    
    # === DAILY INDICATORS ===
    # RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(H4_d[i]) or np.isnan(L4_d[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_d[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ranging = chop_d[i] > 61.8  # Chop > 61.8 indicates ranging market
        
        # Volume confirmation
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # RSI extremes
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        # In ranging markets: fade extreme levels (H4/L4)
        if ranging:
            # Fade H4 (sell at resistance)
            if (close[i] >= H4_d[i] and rsi_overbought and strong_volume and position != -1):
                position = -1
                signals[i] = -0.25
            # Fade L4 (buy at support)
            elif (close[i] <= L4_d[i] and rsi_oversold and strong_volume and position != 1):
                position = 1
                signals[i] = 0.25
            # Exit when price returns to pivot or RSI normalizes
            elif position == 1 and (close[i] >= Pivot_d[i] or rsi[i] < 50):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] <= Pivot_d[i] or rsi[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        
        # In trending markets: breakout of H4/L4 with momentum
        else:
            # Breakout above H4 (buy breakout)
            if (close[i] > H4_d[i] and rsi[i] > 50 and strong_volume and position != 1):
                position = 1
                signals[i] = 0.25
            # Breakdown below L4 (sell breakdown)
            elif (close[i] < L4_d[i] and rsi[i] < 50 and strong_volume and position != -1):
                position = -1
                signals[i] = -0.25
            # Exit when price returns to H3/L3 or momentum fades
            elif position == 1 and (close[i] <= H3_d[i] or rsi[i] < 40):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] >= L3_d[i] or rsi[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals