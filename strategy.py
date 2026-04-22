#!/usr/bin/env python3
"""
Hypothesis: 4-hour Triple EMA with 12-hour VWAP and Volume Spike Filter.
Long when fast EMA > medium EMA > slow EMA and price > 12h VWAP with volume spike.
Short when fast EMA < medium EMA < slow EMA and price < 12h VWAP with volume spike.
Exit when EMA alignment breaks or VWAP condition reverses.
Triple EMA provides robust trend filtering with low whipsaw; 12h VWAP adds institutional context;
volume spike confirms participation. Designed for low trade frequency (<40/year) by requiring
strong confluence. Works in bull markets via trend following and bear markets via short signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Triple EMA: fast (9), medium (21), slow (55)
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Load 12h data for VWAP - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate VWAP on 12h data: typical price * volume cumulative / volume cumulative
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    volume_12h = df_12h['volume'].values
    vwap_12h = np.full_like(typical_price_12h, np.nan)
    
    cum_vol = 0.0
    cum_pv = 0.0
    for i in range(len(typical_price_12h)):
        cum_vol += volume_12h[i]
        cum_pv += typical_price_12h[i] * volume_12h[i]
        if cum_vol > 0:
            vwap_12h[i] = cum_pv / cum_vol
    
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Volume confirmation: current volume > 2.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(55, n):
        # Skip if data not ready
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema55[i]) or 
            np.isnan(vwap_12h_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.5 * vol_ma_30[i]
        
        if position == 0:
            # Long: EMA alignment bullish and price > VWAP with volume spike
            if (ema9[i] > ema21[i] > ema55[i] and 
                close[i] > vwap_12h_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: EMA alignment bearish and price < VWAP with volume spike
            elif (ema9[i] < ema21[i] < ema55[i] and 
                  close[i] < vwap_12h_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: EMA alignment breaks or VWAP condition reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: EMA alignment breaks or price <= VWAP
                if not (ema9[i] > ema21[i] > ema55[i]) or close[i] <= vwap_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: EMA alignment breaks or price >= VWAP
                if not (ema9[i] < ema21[i] < ema55[i]) or close[i] >= vwap_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_TripleEMA_12hVWAP_VolumeSpike"
timeframe = "4h"
leverage = 1.0