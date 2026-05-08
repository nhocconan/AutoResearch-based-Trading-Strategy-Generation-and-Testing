#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Keltner_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA200 trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_1d = (close_1d > ema200_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 1d ATR for Keltner channels (20-period)
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA20 for Keltner center line
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Keltner channels: upper = EMA20 + 2*ATR, lower = EMA20 - 2*ATR
    keltner_upper = ema20_1d_aligned + 2 * atr_1d_aligned
    keltner_lower = ema20_1d_aligned - 2 * atr_1d_aligned
    
    # Choppiness index on 4h (14-period)
    tr_4h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Keltner upper in choppy market (mean reversion) with uptrend bias
            long_cond = (close[i] > keltner_upper[i] and chop[i] > 61.8 and vol_spike[i] and trend_1d_aligned[i] > 0.5)
            # Short: price breaks below Keltner lower in choppy market with downtrend bias
            short_cond = (close[i] < keltner_lower[i] and chop[i] > 61.8 and vol_spike[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Keltner upper (mean reversion)
            if close[i] < keltner_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Keltner lower (mean reversion)
            if close[i] > keltner_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: In choppy markets (Chop > 61.8), price tends to revert to mean after touching Keltner bands (EMA20 ± 2*ATR).
# Uses 1d EMA200 for trend filter to avoid counter-trend trades. Volume spike confirms breakout validity.
# Designed for low frequency (15-30 trades/year) to minimize fee drag while capturing mean reversion in both bull/bear markets.