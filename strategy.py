#!/usr/bin/env python3
name = "6h_Keltner_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop for trend and Keltner
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(10) for Keltner Channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Daily EMA(20) as middle line
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channels: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper = ema_20_1d + 2 * atr_10
    keltner_lower = ema_20_1d - 2 * atr_10
    
    # Align Keltner levels to 6h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Daily trend filter: EMA(50) direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 10, 4)  # Wait for EMA50, ATR10, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or 
            np.isnan(keltner_lower_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or breaks below lower Keltner with volume in daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] <= keltner_lower_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price touches or breaks above upper Keltner with volume in daily downtrend
            elif close[i] >= keltner_upper_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to middle EMA or volume drops
            if close[i] >= ema_20_1d_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to middle EMA or volume drops
            if close[i] <= ema_20_1d_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Keltner Channel reversal with 1d trend and volume confirmation
# - Keltner Channels (EMA20 ± 2*ATR10) on daily timeframe identify dynamic support/resistance
# - In uptrends, price often pulls back to lower Keltner before continuing up (long setup)
# - In downtrends, price often rallies to upper Keltner before continuing down (short setup)
# - Volume spike (1.8x average) confirms institutional participation at these key levels
# - Daily EMA50 trend filter ensures we trade with the higher timeframe trend
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
# - Exit when price returns to EMA20 (middle) or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Novel combination: Keltner reversal (1d) + trend (1d) + volume (6h) - not recently tried on 6h
# - Uses actual daily Keltner levels for better adaptation to volatility
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits