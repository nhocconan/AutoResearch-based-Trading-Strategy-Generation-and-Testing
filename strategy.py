#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirmation
Hypothesis: 1h breakout above/below 4h Camarilla R1/S1 levels in direction of 4h EMA50 trend, confirmed by volume spike (>1.8x 20-bar MA). Camarilla levels provide strong intraday support/resistance. Trend filter ensures alignment with higher timeframe momentum. Volume confirmation reduces false breakouts. Designed for 15-37 trades/year (60-150 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 4h trend while using Camarilla structure for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar (OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R1, S1 levels (based on previous 4h bar's range)
    camarilla_r1 = close_4h + (high_4h - low_4h) * 1.1 / 12
    camarilla_s1 = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (no additional delay needed as they're based on completed 4h bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20  # Position size
    
    # Warmup: max of calculations (20 for vol, 50 for ema)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_4h_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 4h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_4h = close_val > ema_50_val
        bearish_4h = close_val < ema_50_val
        
        # Entry conditions: breakout of Camarilla level in trend direction with volume
        long_entry = (close_val > camarilla_r1_val) and bullish_4h and vol_spike
        short_entry = (close_val < camarilla_s1_val) and bearish_4h and vol_spike
        
        # Exit conditions: opposite Camarilla level touch (or trend reversal)
        exit_long = (close_val < camarilla_s1_val) or not bullish_4h
        exit_short = (close_val > camarilla_r1_val) or not bearish_4h
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0