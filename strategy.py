#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_ChoppyRegime
# Hypothesis: Uses TRIX momentum with volume spikes and choppiness regime filter for robust trend following.
# TRIX filters noise, volume spikes confirm momentum, choppiness regime avoids whipsaws in sideways markets.
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year). Works in bull/bear by adapting to market regime.
# Position size 0.25 for balanced risk management.

name = "4h_TRIX_VolumeSpike_ChoppyRegime"
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
    
    # Calculate TRIX (1-period ROC of triple-smoothed EMA)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change(1).values * 100  # 1-period ROC in percentage
    
    # Calculate ATR for volatility and choppiness
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: measures whether market is choppy (range-bound) or trending
    # High CHOP (>61.8) = ranging, Low CHOP (<38.2) = trending
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume average for confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 34)  # Warmup for chop, volume MA, and daily EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Market regime filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: TRIX positive + volume confirmation + trending regime + daily uptrend
            if trix[i] > 0 and volume_confirm and trending_regime and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX negative + volume confirmation + trending regime + daily downtrend
            elif trix[i] < 0 and volume_confirm and trending_regime and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative or regime turns choppy or daily trend turns down
            if trix[i] < 0 or not trending_regime or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive or regime turns choppy or daily trend turns up
            if trix[i] > 0 or not trending_regime or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals