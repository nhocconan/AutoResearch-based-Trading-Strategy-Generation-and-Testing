#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with volume confirmation and 1d EMA34 trend filter
# Long when price breaks above Camarilla R4 + volume > 1.5x 20-period avg + price > 1d EMA34
# Short when price breaks below Camarilla S4 + volume > 1.5x 20-period avg + price < 1d EMA34
# Uses 4h Camarilla pivot levels for structure and 1d EMA for trend alignment
# Designed for low trade frequency (20-40/year) to minimize fee drag while capturing strong trends
# Works in both bull and bear markets by requiring volume confirmation and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous 4h bar) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_4h + low_4h + close_4h) / 3.0
    range_hl = high_4h - low_4h
    
    # Camarilla levels: R4 = close + (range * 1.1/2), S4 = close - (range * 1.1/2)
    camarilla_r4 = close_4h + (range_hl * 1.1 / 2)
    camarilla_s4 = close_4h - (range_hl * 1.1 / 2)
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    # === 1d Indicators: EMA34 for Trend Filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Camarilla R4
        # 2. Volume confirmation
        # 3. Price above 1d EMA34 (uptrend filter)
        if (close[i] > camarilla_r4_aligned[i]) and vol_confirm and (close[i] > ema_34_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Camarilla S4
        # 2. Volume confirmation
        # 3. Price below 1d EMA34 (downtrend filter)
        elif (close[i] < camarilla_s4_aligned[i]) and vol_confirm and (close[i] < ema_34_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R4S4_Volume_1dEMA34_Filter_v1"
timeframe = "4h"
leverage = 1.0