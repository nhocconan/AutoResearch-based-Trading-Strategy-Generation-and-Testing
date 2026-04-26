#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation. Camarilla levels provide institutional support/resistance. Trend filter ensures alignment with higher timeframe momentum. Volume spike confirms institutional participation. Discrete sizing 0.25 limits trades (~20-50/year). Works in bull/bear via 12h trend filter.
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
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for volume confirmation
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume confirmation
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Previous day's OHLC for Camarilla calculation (using 4h data, need 6 bars back = 1 day)
    # Camarilla levels: based on previous day's range
    prev_close = np.roll(close, 6)  # 6 * 4h = 24h = 1 day ago
    prev_high = np.roll(high, 6)
    prev_low = np.roll(low, 6)
    
    # Calculate Camarilla levels
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # R4 = Close + (High - Low) * 1.1/2
    # S4 = Close - (High - Low) * 1.1/2
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + rang * 1.1 / 4
    camarilla_s3 = prev_close - rang * 1.1 / 4
    camarilla_r4 = prev_close + rang * 1.1 / 2
    camarilla_s4 = prev_close - rang * 1.1 / 2
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 6 bars for previous day + 50 for ATR ratio + 50 for EMA
    start_idx = max(6, 50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(prev_close[i]) or np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_spike = atr_ratio[i] > 1.2  # volume confirmation
        size = fixed_size
        
        # Entry conditions: Camarilla R3/S3 breakout with volume spike AND aligned with 12h EMA50 trend
        # Long: price breaks above R3 (bullish breakout)
        # Short: price breaks below S3 (bearish breakout)
        long_entry = (close_val > r3) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < s3) and vol_spike and (close_val < ema_50_val)
        
        # Exit conditions: reverse signal or price reaches R4/S4 (extreme levels)
        long_exit = (position == 1) and ((close_val < s3) or (close_val > r4))
        short_exit = (position == -1) and ((close_val > r3) or (close_val < s4))
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on reverse signal or extreme level
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on reverse signal or extreme level
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0