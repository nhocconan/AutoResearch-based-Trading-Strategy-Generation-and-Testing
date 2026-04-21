#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_Volume_ChopRegime_ATRStop_V1
Hypothesis: 4h Camarilla R3/S3 breakout with volume confirmation (>1.5x 20-period volume MA) and choppiness regime filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Uses 12h HTF for trend filter (price > EMA34 for longs, < EMA34 for shorts). ATR-based stoploss via signal=0 when price moves against position by 2.5*ATR. Designed for low trade frequency (target: 75-150 total 4h trades over 4 years) to minimize fee drag and work in both bull/bear markets via regime adaptation. Focus on BTC/ETH as primary targets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    close_4h_series = pd.Series(close_4h)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    
    # Calculate Camarilla pivot levels from previous day
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Align daily data to 4h timeframe
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_4h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels: R3, S3 (most significant for breakouts)
    # R3 = Close + 1.1*(High-Low)*1.1/4
    # S3 = Close - 1.1*(High-Low)*1.1/4
    camarilla_range = prev_high_4h - prev_low_4h
    camarilla_r3 = prev_close_4h + 1.1 * camarilla_range * 1.1 / 4
    camarilla_s3 = prev_close_4h - 1.1 * camarilla_range * 1.1 / 4
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    chop_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = high_4h_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_4h_series.rolling(window=14, min_periods=14).min().values
    denominator = highest_high - lowest_low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(chop_sum / denominator) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i])
            or np.isnan(ema_34_12h_aligned[i])):
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
            # Long: Camarilla R3 breakout + volume + trend filter (in uptrend or choppy market)
            if price > camarilla_r3[i] and vol_ok and (price > ema_34_12h_aligned[i] or is_choppy):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Camarilla S3 breakdown + volume + trend filter (in downtrend or choppy market)
            elif price < camarilla_s3[i] and vol_ok and (price < ema_34_12h_aligned[i] or is_choppy):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Camarilla S3 retest or loss of volume/momentum
            elif price < camarilla_s3[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Camarilla R3 retest or loss of volume/momentum
            elif price > camarilla_r3[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_Volume_ChopRegime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0