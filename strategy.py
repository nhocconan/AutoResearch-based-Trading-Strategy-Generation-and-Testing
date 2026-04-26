#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v2
Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA50 trend filter and ATR(14)>ATR(50) volume spike filter.
- Long when price breaks above Camarilla R3 AND 1w EMA50 uptrend AND ATR(14) > ATR(50) (vol expansion)
- Short when price breaks below Camarilla S3 AND 1w EMA50 downtrend AND ATR(14) > ATR(50)
- Uses Camarilla pivot levels from completed 1d bars (structure) with 1w trend filter (HTF) to avoid counter-trend whipsaws
- ATR-based volatility expansion filter ensures institutional participation and reduces false breakouts
- Designed for moderate frequency (target 12-37 trades/year) to minimize fee drag
- Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or trend reversal
- Novelty: Combines Camarilla structure with weekly trend and volatility expansion for robust edge in bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior 1d bar (completed bar only)
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of prior day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (no additional delay needed for structure)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter (needs completed weekly candle)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate ATR filter: ATR(14) > ATR(50) for volatility expansion
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_expansion = atr_14 > atr_50  # Volatility expansion filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for ATR, 50 for EMA)
    start_idx = max(50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_1w[i]) or np.isnan(atr_expansion[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with trend and volatility expansion filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND weekly uptrend AND vol expansion
            if close[i] > camarilla_r3_aligned[i] and trend_1w[i] == 1 and atr_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND weekly downtrend AND vol expansion
            elif close[i] < camarilla_s3_aligned[i] and trend_1w[i] == -1 and atr_expansion[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR weekly trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR weekly trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0