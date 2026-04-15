#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with volume confirmation and 1d EMA34 trend filter
# Long when price breaks above 1d Camarilla R4 + volume > 1.5x 20-period avg + close > 1d EMA34
# Short when price breaks below 1d Camarilla S4 + volume > 1.5x 20-period avg + close < 1d EMA34
# Uses tighter R4/S4 levels (vs R3/S3) for fewer, higher-quality signals.
# EMA34 filter ensures we trade with the intermediate-term trend, avoiding counter-trend whipsaws.
# Designed for low trade frequency (15-25/year) to minimize fee drag on 12h timeframe.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by requiring alignment with 1d EMA34.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R4, S4) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R4 = close + (high - low) * 1.1/2, S4 = close - (high - low) * 1.1/2
    camarilla_r4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # === 1d Indicator: EMA34 (trend filter) ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema34_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R4 level
        # 2. Uptrend (close > 1d EMA34)
        # 3. Volume confirmation
        if (close[i] > camarilla_r4_aligned[i]) and \
           (close[i] > ema34_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S4 level
        # 2. Downtrend (close < 1d EMA34)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s4_aligned[i]) and \
             (close[i] < ema34_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_CamarillaR4S4_Volume_EMA34_Filter_v1"
timeframe = "12h"
leverage = 1.0