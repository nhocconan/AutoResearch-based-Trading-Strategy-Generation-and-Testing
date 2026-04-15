#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly trend filter and volume confirmation
# Long when price breaks above Camarilla R4 (1d) + volume > 2.0x 20-period avg + price > weekly EMA50
# Short when price breaks below Camarilla S4 (1d) + volume > 2.0x 20-period avg + price < weekly EMA50
# Uses 1d Camarilla pivots for structure, weekly EMA for trend alignment, and volume for confirmation
# Designed for low trade frequency (12-25/year) to minimize fee drag while capturing strong breakouts
# Weekly trend filter ensures we only trade with the higher timeframe momentum
# Works in both bull and bear markets by requiring volume confirmation and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (using previous day's OHLC) ===
    # Camarilla levels calculated from previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 6h timeframe (wait for 1d bar to close)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # === 1w Indicators: EMA50 for Trend Filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R4 level
        # 2. Volume confirmation
        # 3. Price above weekly EMA50 (uptrend filter)
        if (close[i] > R4_aligned[i]) and vol_confirm and (close[i] > ema_50_1w_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S4 level
        # 2. Volume confirmation
        # 3. Price below weekly EMA50 (downtrend filter)
        elif (close[i] < S4_aligned[i]) and vol_confirm and (close[i] < ema_50_1w_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_Volume_1wEMA50_Filter_v1"
timeframe = "6h"
leverage = 1.0