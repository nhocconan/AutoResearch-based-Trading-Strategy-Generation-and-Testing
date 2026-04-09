#!/usr/bin/env python3
# 12h_hma_vol_chop_regime_v2
# Hypothesis: 12h strategy using 1d HMA(21) for trend filter, volume confirmation (>1.3x 20-bar average volume), and 1d chop regime filter (chop > 61.8 = range, chop < 38.2 = trending). Enters long in uptrend + volume + chop<38.2; enters short in downtrend + volume + chop<38.2. Exits on trend reversal or chop>61.8. Uses discrete sizing (0.25) to minimize fee drag. Target: 12-37 trades/year. Works in bull/bear by following established trend with volume confirmation in trending regimes only.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_hma_vol_chop_regime_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = 10 days of 12h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d HMA(21) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate HMA(21) on 1d close
    n_hma = 21
    half_n = n_hma // 2
    sqrt_n = int(np.sqrt(n_hma))
    
    wma_half = pd.Series(close_1d).rolling(window=half_n, min_periods=half_n).mean().values
    wma_full = pd.Series(close_1d).rolling(window=n_hma, min_periods=n_hma).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_1d = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Multi-timeframe: 1d Chopiness Index (14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_series = pd.Series(close_1d)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    
    # Chopiness Index = 100 * log15(sum(TR14)/(ATR14*14)) / log15(14)
    sum_tr_14 = tr.rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop = np.where(atr_14 > 0, chop_raw, 50.0)  # default to 50 when ATR=0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Trend filters
        uptrend = close[i] > hma_21_1d_aligned[i]
        downtrend = close[i] < hma_21_1d_aligned[i]
        
        # Regime filter: chop < 38.2 = trending (good for trend following)
        trending_regime = chop_aligned[i] < 38.2
        choppy_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: trend turns down OR market gets choppy
            if not uptrend or choppy_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns up OR market gets choppy
            if not downtrend or choppy_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only in trending regime with volume confirmation
            if trending_regime and volume_confirmed:
                if uptrend:
                    position = 1
                    signals[i] = 0.25
                elif downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals