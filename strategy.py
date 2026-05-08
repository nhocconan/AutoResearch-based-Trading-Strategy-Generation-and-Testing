#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KeltnerBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once for ATR and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w ATR(14) for Keltner channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1w EMA20 for middle line and trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20_1w + (2.0 * atr_1w)
    lower_keltner = ema20_1w - (2.0 * atr_1w)
    
    # Align Keltner channels to daily timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume spike: current volume > 1.5 * 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper Keltner with volume spike
            long_cond = (close[i] > upper_keltner_aligned[i] and vol_spike[i])
            
            # Short entry: price breaks below lower Keltner with volume spike
            short_cond = (close[i] < lower_keltner_aligned[i] and vol_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA20 (trend reversal)
            if close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA20 (trend reversal)
            if close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Keltner channel breakout strategy on daily timeframe with 1-week trend filter.
# Enters long when price breaks above upper Keltner (EMA20 + 2*ATR) with volume spike.
# Enters short when price breaks below lower Keltner (EMA20 - 2*ATR) with volume spike.
# Exits when price crosses back below/above the 1-week EMA20.
# Uses weekly ATR and EMA for stability, daily price action for entry timing.
# Volume spike filters out low-conviction breakouts. Designed for 10-20 trades/year.
# Works in bull markets (breakouts with momentum) and bear markets (mean reversion from extremes).