#!/usr/bin/env python3
# Strategy: 4h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter
# Hypothesis: Breakout above daily Camarilla R1 or below S1 with volume confirmation and 1d EMA34 trend filter on 4h timeframe.
# Uses 4h bars for entries, filtering by 1d trend to avoid counter-trend trades. Volume > 2x 20-period MA confirms institutional interest.
# ATR-based stoploss limits drawdown. Designed for 20-50 trades/year to minimize fee drag and work in both bull/bear markets.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis (trend, pivot levels)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = pivot_1d + (range_1d * 1.1 / 12)
    S1 = pivot_1d - (range_1d * 1.1 / 12)
    
    # Align 1d indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Load 4h data for entry timing, volume, ATR
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume spike detection (20-period on 4h)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # ATR for volatility filter (14-period on 4h)
    high_low = high_4h - low_4h
    high_close = np.abs(high_4h - np.roll(close_4h, 1))
    low_close = np.abs(low_4h - np.roll(close_4h, 1))
    high_low[0] = high_4h[0] - low_4h[0]
    high_close[0] = np.abs(high_4h[0] - close_4h[0])
    low_close[0] = np.abs(low_4h[0] - close_4h[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        
        if position == 0:
            # Long: price breaks above R1, above 1d EMA34 (uptrend), with volume confirmation
            if (price > R1_aligned[i] and 
                price > ema34_1d_aligned[i] and 
                vol > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 1d EMA34 (downtrend), with volume confirmation
            elif (price < S1_aligned[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or ATR-based stop
            if (price < S1_aligned[i] or 
                price < low_4h[i] - 1.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or ATR-based stop
            if (price > R1_aligned[i] or 
                price > high_4h[i] + 1.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0