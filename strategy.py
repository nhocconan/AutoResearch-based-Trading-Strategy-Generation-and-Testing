#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTFTrend_ATRStop_V1
Hypothesis: 4h Donchian(20) breakout with 12h HTF trend filter (HMA21) and volume confirmation (>1.5x 20-period volume MA) with ATR-based trailing stop. 
Donchian channels provide clear breakout levels, HTF HMA filters for higher-timeframe trend alignment to avoid counter-trend trades, 
volume confirmation reduces false breakouts, and ATR stoploss manages risk. 
Target 19-50 trades/year (75-200 total over 4 years) for BTC/ETH/SOL.
Uses 4h primary timeframe with 12h HTF for trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for HMA trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # === 12h HMA21 for trend filter ===
    close_12h = df_12h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 12 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).mean().values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian(20) channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high_4h - low_4h).values
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1))).values
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1))).values
    tr2[0] = tr1[0]  # first bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(hma_21_12h_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + 12h uptrend (price > HMA)
            if price > donch_high[i] and vol_ok and price > hma_21_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below Donchian low + volume confirmation + 12h downtrend (price < HMA)
            elif price < donch_low[i] and vol_ok and price < hma_21_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest since entry
            if price < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest since entry
            if price > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTFTrend_ATRStop_V1"
timeframe = "4h"
leverage = 1.0