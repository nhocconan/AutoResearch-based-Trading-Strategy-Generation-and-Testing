#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_Volume_ATRFilter
Hypothesis: 6h Camarilla pivot breakout at R1/S1 with volume confirmation (>1.2x 20-period volume MA) and ATR-based trend filter (ATR(14) > ATR(50) for volatility expansion). Uses 1d HTF for trend filter (price > EMA50 for longs, < EMA50 for shorts). ATR-based stoploss via signal=0 when price moves against position by 1.5*ATR. Designed for low trade frequency (<150 total 6h trades) to minimize fee drag and work in both bull/bear markets via volatility regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    r4 = pivot + (range_hl * 1.1 / 2.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss and volatility filter
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(r4[i]) or np.isnan(s4[i])
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])
            or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.2 * vol_ma[i]  # volume confirmation
        vol_expansion = atr[i] > atr_ma[i]  # volatility expansion filter
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility expansion, aligned with 1d uptrend
            if price > r1[i] and vol_ok and vol_expansion and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume and volatility expansion, aligned with 1d downtrend
            elif price < s1[i] and vol_ok and vol_expansion and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price reaches R4 (take profit) or breaks below S1 (reversal)
            elif price >= r4[i] or price < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price reaches S4 (take profit) or breaks above R1 (reversal)
            elif price <= s4[i] or price > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0