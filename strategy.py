#!/usr/bin/env python3
"""
4h_HTF_Pivot_R3S3_Breakout_VolumeATRFilter_V1
Hypothesis: Breakout above 1d Camarilla R3 or below S3 with volume confirmation (>1.8x 20-bar MA) and ATR stoploss (2.5x) works on 4h timeframe. Uses 1d for pivot calculation and 12h for trend filter (EMA34). Tight entries target 20-50 trades/year per symbol. Works in bull/bear via mean-reversion exits at opposite Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')  # for EMA34 trend filter
    
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R3/S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    camarilla_r3 = prev_close + range_1d * 1.1 / 4  # R3 = close + 1.1*range/4
    camarilla_s3 = prev_close - range_1d * 1.1 / 4  # S3 = close - 1.1*range/4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 12h EMA34 for Trend Filter ===
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === 4h Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume MA (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.8 * vol_ma[i]  # strict volume filter to reduce trades
        trend_up = ema_12h_aligned[i] < price  # price above 12h EMA34 = uptrend
        trend_down = ema_12h_aligned[i] > price  # price below 12h EMA34 = downtrend
        
        if position == 0:
            # Long: break above R3 with volume and uptrend
            if price > camarilla_r3_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and downtrend
            elif price < camarilla_s3_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: mean reversion to S3 or ATR stoploss
            if price < camarilla_s3_aligned[i] or price < close[i-1] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: mean reversion to R3 or ATR stoploss
            if price > camarilla_r3_aligned[i] or price > close[i-1] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Pivot_R3S3_Breakout_VolumeATRFilter_V1"
timeframe = "4h"
leverage = 1.0