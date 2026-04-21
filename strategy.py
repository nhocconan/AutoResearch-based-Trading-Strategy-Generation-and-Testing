#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V3
Hypothesis: Daily Camarilla pivot R1/S1 breakouts with volume confirmation and choppiness regime filter capture institutional breakout/retest patterns. Works in both bull/bear markets: regime filter adapts to trending/choppy conditions, volume confirms institutional participation, ATR stop manages risk. Target: 7-25 trades/year (30-100 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for trend filter (HTF) ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d indicators (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Camarilla Pivot Points (from previous day) ===
    # Calculate from previous day's OHLC
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # first day has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === Choppiness Index regime filter (14-period) ===
    # CHOP = 100 * log10(sum(TR over period) / (ATR * period)) / log10(period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    
    # === ATR (20-period) for stoploss ===
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(chop[i]) or np.isnan(atr_20[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol_ratio = volume_1d[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Regime adaptive: CHOP > 50 = choppy (mean revert), CHOP < 50 = trending (breakout)
            if chop[i] > 50:
                # Choppy regime: mean reversion at S1/R1
                # Long: price crosses above S1 with volume
                if price > s1[i] and close_1d[i-1] <= s1[i] and vol_ratio > 1.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: price crosses below R1 with volume
                elif price < r1[i] and close_1d[i-1] >= r1[i] and vol_ratio > 1.5:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:
                # Trending regime: breakout of R1/S1 with volume
                # Long: price breaks above R1 with volume
                if price > r1[i] and close_1d[i-1] <= r1[i] and vol_ratio > 2.0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: price breaks below S1 with volume
                elif price < s1[i] and close_1d[i-1] >= s1[i] and vol_ratio > 2.0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr_20[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: opposite Camarilla level touch or regime shift
            elif price >= r1[i] or (chop[i] > 60 and price <= pivot[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr_20[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: opposite Camarilla level touch or regime shift
            elif price <= s1[i] or (chop[i] > 60 and price >= pivot[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V3"
timeframe = "1d"
leverage = 1.0