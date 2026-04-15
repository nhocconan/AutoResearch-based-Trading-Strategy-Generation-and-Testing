#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume confirmation and 1w trend filter
# Long when price breaks above 1d Camarilla R1 + 1d volume > 1.3x 20-period volume SMA + 1w close > 1w EMA20
# Short when price breaks below 1d Camarilla S1 + 1d volume > 1.3x 20-period volume SMA + 1w close < 1w EMA20
# Uses discrete position sizing (0.25) to minimize fee drag. Designed for 12-37 trades/year on 12h timeframe.
# Camarilla levels provide precise intraday pivot points, volume confirmation avoids false breakouts,
# and weekly EMA filter ensures alignment with higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    camarilla_r1_1d = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # === 1d Indicators: Volume Confirmation ===
    vol_sma_20_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_sma_20_1d * 1.3)
    
    # === 1w Indicators: EMA20 for Trend Filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_sma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_ok = volume[i] > (vol_sma_20_1d[i] * 1.3)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. 1w close above 1w EMA20 (uptrend filter)
        if (close[i] > camarilla_r1_aligned[i]) and vol_ok and (close_1w[-1] > ema_20_1w_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. 1w close below 1w EMA20 (downtrend filter)
        elif (close[i] < camarilla_s1_aligned[i]) and vol_ok and (close_1w[-1] < ema_20_1w_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1wEMA_Filter_v1"
timeframe = "12h"
leverage = 1.0