#!/usr/bin/env python3
"""
6h_EMA_Crossover_Volume_Regime
Hypothesis: 6s trend following using EMA crossovers (21/55) with volume confirmation and regime filter (ADX < 25 for range, > 25 for trend).
Trades only in direction of 12h EMA55 trend to avoid counter-trend whipsaws. Volume spike (>1.5x 20 EMA) confirms breakout strength.
Designed for low trade frequency (~20-50/year) to minimize fee drag. Works in bull by capturing trends, in bear by avoiding false signals during low ADX.
"""

name = "6h_EMA_Crossover_Volume_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Data for Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema55_12h = pd.Series(close_12h).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema55_12h_aligned = align_htf_to_ltf(prices, df_12h, ema55_12h)
    
    # === EMA Crossovers (21/55) on 6h ===
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # === ADX (14) for Regime Filter on 6h ===
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros(len(high))
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth = np.zeros(len(high))
        minus_dm_smooth = np.zeros(len(high))
        plus_dm_smooth[period-1] = np.sum(plus_dm[:period])
        minus_dm_smooth[period-1] = np.sum(minus_dm[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros(len(high))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.zeros(len(high))
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # === Volume Filter: 1.5x 20-period EMA on 6h ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA55 and ADX)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema21[i]) or np.isnan(ema55[i]) or 
            np.isnan(ema55_12h_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: EMA21 crosses above EMA55, uptrend (12h EMA55 rising), ADX > 25, volume spike
            if (ema21[i] > ema55[i] and ema21[i-1] <= ema55[i-1] and
                close[i] > ema55_12h_aligned[i] and
                adx[i] > 25 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: EMA21 crosses below EMA55, downtrend (12h EMA55 falling), ADX > 25, volume spike
            elif (ema21[i] < ema55[i] and ema21[i-1] >= ema55[i-1] and
                  close[i] < ema55_12h_aligned[i] and
                  adx[i] > 25 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: EMA21 crosses below EMA55 or ADX drops below 20 (range)
            if ema21[i] < ema55[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: EMA21 crosses above EMA55 or ADX drops below 20 (range)
            if ema21[i] > ema55[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals