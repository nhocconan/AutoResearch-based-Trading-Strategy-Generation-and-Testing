#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V4
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation (>1.3x 20-period volume MA) and choppiness regime filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Uses 1d HTF for trend filter (price > EMA50 for longs, < EMA50 for shorts). ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR. Designed for low trade frequency (<200 total 4h trades) to minimize fee drag and work in both bull/bear markets via regime adaptation.
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
    
    # Calculate Camarilla levels from previous 4h bar (using daily OHLC)
    # For 4h timeframe, we use daily OHLC to calculate Camarilla levels
    # We'll calculate them once per day using the 1d data
    # But since we need them for 4h bars, we'll calculate from 1d and align
    
    # Get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = np.zeros_like(daily_close)
    camarilla_s1 = np.zeros_like(daily_close)
    camarilla_r2 = np.zeros_like(daily_close)
    camarilla_s2 = np.zeros_like(daily_close)
    camarilla_r3 = np.zeros_like(daily_close)
    camarilla_s3 = np.zeros_like(daily_close)
    camarilla_r4 = np.zeros_like(daily_close)
    camarilla_s4 = np.zeros_like(daily_close)
    camarilla_pp = np.zeros_like(daily_close)
    
    # Only calculate when we have valid data
    valid_idx = ~(np.isnan(daily_high) | np.isnan(daily_low) | np.isnan(daily_close))
    if np.any(valid_idx):
        # Pivot point
        camarilla_pp[valid_idx] = (daily_high[valid_idx] + daily_low[valid_idx] + daily_close[valid_idx]) / 3.0
        
        # Range
        range_val = daily_high[valid_idx] - daily_low[valid_idx]
        
        # Camarilla levels
        camarilla_r1[valid_idx] = camarilla_pp[valid_idx] + range_val * 1.1 / 12.0
        camarilla_s1[valid_idx] = camarilla_pp[valid_idx] - range_val * 1.1 / 12.0
        camarilla_r2[valid_idx] = camarilla_pp[valid_idx] + range_val * 1.1 / 6.0
        camarilla_s2[valid_idx] = camarilla_pp[valid_idx] - range_val * 1.1 / 6.0
        camarilla_r3[valid_idx] = camarilla_pp[valid_idx] + range_val * 1.1 / 4.0
        camarilla_s3[valid_idx] = camarilla_pp[valid_idx] - range_val * 1.1 / 4.0
        camarilla_r4[valid_idx] = camarilla_pp[valid_idx] + range_val * 1.1 / 2.0
        camarilla_s4[valid_idx] = camarilla_pp[valid_idx] - range_val * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_4h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_4h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pp_4h = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
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
    chop = 100 * np.log10(chop_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_4h[i]) or np.isnan(camarilla_s1_4h[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i])
            or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume confirmation
        
        # Regime detection
        is_choppy = chop[i] > 61.8  # mean reversion regime
        is_trending = chop[i] < 38.2  # trend following regime
        
        if position == 0:
            # Long: Camarilla S1 breakout + volume + trend filter (in uptrend or choppy market)
            if price > camarilla_s1_4h[i] and vol_ok and (price > ema_50_1d_aligned[i] or is_choppy):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Camarilla R1 breakdown + volume + trend filter (in downtrend or choppy market)
            elif price < camarilla_r1_4h[i] and vol_ok and (price < ema_50_1d_aligned[i] or is_choppy):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price below S1 or loss of volume/momentum
            elif price < camarilla_s1_4h[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price above R1 or loss of volume/momentum
            elif price > camarilla_r1_4h[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V4"
timeframe = "4h"
leverage = 1.0