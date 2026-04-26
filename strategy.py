#!/usr/bin/env python3
"""
1d_KAMA_Trend_Regime_Filter_v1
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) captures trend with low lag, combined with choppiness regime filter to avoid whipsaws in ranging markets. Uses volume confirmation for breakout strength. Designed for 1d to minimize fee drag while capturing major trends in both bull and bear markets via adaptive trend filter and regime detection. Target 10-25 trades/year.
"""

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
    
    # Load 1w data ONCE before loop for regime filter (optional, can use 1d chop)
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA(10, 2, 30) - adaptive trend indicator
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    change = np.concatenate([np.full(10, np.nan), change])
    
    volatility = np.abs(np.diff(close, n=1))
    volatility = np.concatenate([np.array([np.nan]), volatility])
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.divide(change, volatility_sum, out=np.full_like(change, np.nan), where=volatility_sum!=0)
    
    # Smoothing constants
    fastest = 2.0 / (2 + 1)   # EMA(2)
    slowest = 2.0 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Choppiness Index (CHOP) for regime detection - uses 14 periods
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.full_like(close, np.nan)
    for i in range(13, n):
        if tr_sum[i] > 0 and hh[i] > ll[i]:
            log_val = np.log10(tr_sum[i] / (hh[i] - ll[i]))
            chop[i] = 100 * log_val / np.log10(14)
        else:
            chop[i] = np.nan
    
    # Volume average for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of KAMA seed, CHOP, volume
    start_idx = max(50, 20)  # ensure sufficient data
    
    for i in range(start_idx, n):
        close_val = close[i]
        kama_val = kama[i]
        chop_val = chop[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(chop_val) or np.isnan(avg_vol)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Regime filter: CHOP < 50 = trending (favor trend following), CHOP >= 50 = ranging (avoid)
        trending_regime = chop_val < 50
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long: price > KAMA in trending regime with volume
        long_condition = (close_val > kama_val) and trending_regime and volume_confirmed
        # Short: price < KAMA in trending regime with volume
        short_condition = (close_val < kama_val) and trending_regime and volume_confirmed
        
        # Exit: opposite signal or regime change to ranging
        long_exit = (position == 1 and (close_val <= kama_val or not trending_regime))
        short_exit = (position == -1 and (close_val >= kama_val or not trending_regime))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Trend_Regime_Filter_v1"
timeframe = "1d"
leverage = 1.0