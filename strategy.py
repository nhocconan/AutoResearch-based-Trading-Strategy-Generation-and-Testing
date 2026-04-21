#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation (>1.5x 20-period volume MA) and choppiness regime filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Uses 1d HTF for trend filter (price > EMA50 for longs, < EMA50 for shorts). ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR. Designed for low trade frequency (<150 total 4h trades) to minimize fee drag and work in both bull/bear markets via regime adaptation.
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
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla pivot levels from previous day (using 1d data)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's high, low, close to calculate today's levels
    # Since we're on 4h chart, we'll use the 1d data shifted by 1 to get previous day
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_4h['high'].values, 6)  # Approximate: 6*4h = 1d
    prev_low_1d = np.roll(df_4h['low'].values, 6)
    
    # For simplicity, we'll use rolling window on 4h to approximate daily pivot
    # This is not perfect but avoids look-ahead and uses available data
    # Better approach: use actual 1d OHLC from mtf_data
    # Let's get the actual 1d OHLC properly
    
    # Recalculate using proper 1d data from mtf_data
    # We already have df_1d from get_htf_data
    if len(df_1d) >= 2:
        prev_close_1d = np.roll(df_1d['close'].values, 1)
        prev_high_1d = np.roll(df_1d['high'].values, 1)
        prev_low_1d = np.roll(df_1d['low'].values, 1)
        
        # Calculate Camarilla levels for current day based on previous day
        camarilla_r1 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
        camarilla_s1 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
        
        # Align to 4h timeframe
        camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
        camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    else:
        camarilla_r1_aligned = np.full(n, np.nan)
        camarilla_s1_aligned = np.full(n, np.nan)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    chop_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    chop = 100 * np.log10(chop_sum / hh_ll) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i])
            or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Regime detection
        is_choppy = chop[i] > 61.8  # mean reversion regime
        is_trending = chop[i] < 38.2  # trend following regime
        
        if position == 0:
            # Long: price > Camarilla R1 + volume + trend filter (in uptrend or choppy market)
            if price > camarilla_r1_aligned[i] and vol_ok and (price > ema_50_1d_aligned[i] or is_choppy):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price < Camarilla S1 + volume + trend filter (in downtrend or choppy market)
            elif price < camarilla_s1_aligned[i] and vol_ok and (price < ema_50_1d_aligned[i] or is_choppy):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price < Camarilla S1 or loss of volume/momentum
            elif price < camarilla_s1_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price > Camarilla R1 or loss of volume/momentum
            elif price > camarilla_r1_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0