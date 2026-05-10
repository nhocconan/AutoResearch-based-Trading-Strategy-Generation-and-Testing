#!/usr/bin/env python3
# 4h_KeltnerChannel_Breakout_12hTrend_Volume
# Hypothesis: Long when price breaks above Keltner upper band (EMA20 + 2*ATR) with volume > 1.5x average in uptrend (price > 12h EMA50).
# Short when price breaks below Keltner lower band (EMA20 - 2*ATR) with volume > 1.5x average in downtrend (price < 12h EMA50).
# Exit when price crosses back below/above EMA20 or ATR-based stoploss hit.
# Designed for 20-50 trades/year to avoid fee drag. Works in bull/bear via trend filter.

name = "4h_KeltnerChannel_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for Keltner bands and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # EMA20 for Keltner basis
    ema20 = np.full(n, np.nan)
    k = 2 / (20 + 1)
    for i in range(20, n):
        if i == 20:
            ema20[i] = np.mean(close[0:20])
        else:
            ema20[i] = close[i] * k + ema20[i-1] * (1 - k)
    
    # Keltner bands
    keltner_up = ema20 + 2 * atr
    keltner_dn = ema20 - 2 * atr
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(keltner_up[i]) or np.isnan(keltner_dn[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 12h EMA50 trend
            if close[i] > ema_50_12h_aligned[i]:  # Uptrend
                # Long: Breakout above Keltner upper band with volume confirmation
                if close[i] > keltner_up[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Breakout below Keltner lower band with volume confirmation
                if close[i] < keltner_dn[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA20 or stoploss hit
            if close[i] < ema20[i] or (i > 0 and low[i] < keltner_dn[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above EMA20 or stoploss hit
            if close[i] > ema20[i] or (i > 0 and high[i] > keltner_up[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals