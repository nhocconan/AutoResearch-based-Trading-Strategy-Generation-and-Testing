#!/usr/bin/env python3
name = "1d_TRIX_VolumeSpike_ChopRegime"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # TRIX on weekly close (15-period EMA of EMA of EMA)
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Align TRIX to daily timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # Weekly EMA(50) for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume spike detection: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index on daily (14-period)
    def choppiness_index(high, low, close, window=14):
        atr = []
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        tr = np.concatenate([[np.nan], tr])
        for i in range(len(tr)):
            if i < window:
                atr.append(np.nan)
            else:
                atr.append(np.nanmean(tr[i-window+1:i+1]))
        atr = np.array(atr)
        highest_high = pd.Series(high).rolling(window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window, min_periods=window).min().values
        chop = np.where(
            (highest_high - lowest_low) > 0,
            100 * np.log10(np.nansum(atr[i-window+1:i+1]) / np.log2(window) / (highest_high - lowest_low)),
            50
        )
        return chop
    
    chop = choppiness_index(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX positive + volume spike + trending market (CHOP < 38.2)
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            trending = chop[i] < 38.2
            uptrend = ema50_1w_aligned[i] > ema50_1w_aligned[i-1]
            
            if trix_aligned[i] > 0 and vol_condition and trending and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative + volume spike + trending market (CHOP < 38.2)
            elif trix_aligned[i] < 0 and vol_condition and trending and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX turns negative or chop increases (range market)
            if trix_aligned[i] < 0 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX turns positive or chop increases (range market)
            if trix_aligned[i] > 0 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX momentum on weekly timeframe with volume spike and chop regime filter
# - TRIX (triple EMA crossover) captures momentum with less lag than MACD
# - Weekly TRIX > 0 = bullish momentum, < 0 = bearish momentum
# - Volume spike (2x average) confirms institutional participation in the move
# - Chop regime filter: only trade when CHOP < 38.2 (trending market), avoid when CHOP > 61.8 (range)
# - Works in bull markets (buy TRIX>0 with volume in uptrend) and bear markets (sell TRIX<0 with volume in downtrend)
# - Exit when momentum reverses or market becomes ranging
# - Position size 0.25 targets 20-40 trades/year, avoiding fee drag
# - Weekly timeframe reduces noise, daily execution improves timing