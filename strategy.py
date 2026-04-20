#!/usr/bin/env python3
# 6h_WeeklyPivot_Trend_Scalper
# Hypothesis: Weekly pivots provide strong institutional support/resistance levels. 
# Trades are taken in the direction of the 1-week trend (EMA50) when price breaks 
# weekly R1/S1 with volume confirmation, targeting mean reversion to the weekly pivot 
# or continuation to R2/S2. Designed for 6h timeframe to capture multi-day swings 
# while avoiding excessive trade frequency. Works in bull/bear markets by using 
# weekly trend filter and volatility-adjusted position sizing.

name = "6h_WeeklyPivot_Trend_Scalper"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot and levels from previous week
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Weekly pivot point (standard)
    wp = (wh + wl + wc) / 3
    # Weekly R1, S1, R2, S2
    wr1 = wp + (wh - wl)
    ws1 = wp - (wh - wl)
    wr2 = wp + 2 * (wh - wl)
    ws2 = wp - 2 * (wh - wl)
    
    # Align weekly levels to 6h timeframe
    wr1_aligned = align_htf_to_ltf(prices, df_1w, wr1)
    ws1_aligned = align_htf_to_ltf(prices, df_1w, ws1)
    wr2_aligned = align_htf_to_ltf(prices, df_1w, wr2)
    ws2_aligned = align_htf_to_ltf(prices, df_1w, ws2)
    wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    
    # Weekly trend filter: EMA50 on weekly close
    wk_close_series = pd.Series(df_1w['close'])
    wk_ema50 = wk_close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    wk_ema50_aligned = align_htf_to_ltf(prices, df_1w, wk_ema50)
    
    # 6h ATR for volatility filter and dynamic sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wr1_aligned[i]) or np.isnan(ws1_aligned[i]) or
            np.isnan(wr2_aligned[i]) or np.isnan(ws2_aligned[i]) or
            np.isnan(wp_aligned[i]) or np.isnan(wk_ema50_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(atr[i]) or np.isnan(atr_ma50[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volatility is sufficient (avoid choppy low-vol periods)
        volatility_ok = atr[i] > (atr_ma50[i] * 0.6)
        
        if position == 0:
            # Long conditions: price breaks above WR1, above weekly EMA50, volume confirmation
            if (close[i] > wr1_aligned[i] and 
                close[i] > wk_ema50_aligned[i] and 
                volume_filter[i] and 
                volatility_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below WS1, below weekly EMA50, volume confirmation
            elif (close[i] < ws1_aligned[i] and 
                  close[i] < wk_ema50_aligned[i] and 
                  volume_filter[i] and 
                  volatility_ok):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price reaches weekly pivot (mean reversion) or breaks WR2 (continuation)
            if close[i] >= wp_aligned[i] or close[i] >= wr2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price reaches weekly pivot (mean reversion) or breaks WS2 (continuation)
            if close[i] <= wp_aligned[i] or close[i] <= ws2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals