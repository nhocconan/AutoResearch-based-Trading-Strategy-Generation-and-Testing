#!/usr/bin/env python3
"""
6h_trix_1d_regime_volume_v1
Hypothesis: TRIX momentum combined with 1d Choppiness Index regime filter and volume confirmation captures trend persistence while avoiding choppy markets. Works in bull markets via trend continuation and in bear markets via mean reversion in range regimes. Targets 15-35 trades/year with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_trix_1d_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for regime and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # TRIX (15-period) on 6h close
    ema1 = pd.Series(close).ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # 1d Choppiness Index (14-period) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    
    chop_denom = np.log14(14) * atr_ma_1d
    chop = np.where(chop_denom != 0, 
                    100 * np.log10(np.sum(atr_1d[-14:] if len(atr_1d) >= 14 else atr_1d) / chop_denom) / np.log10(14), 
                    50)
    # Simplified chop calculation using rolling sum
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    log14 = np.log(14) / np.log(10)  # log base 10 of 14
    chop = 100 * np.log10(atr_sum / (log14 * atr_ma_1d)) / np.log10(14) if 'log10' in dir(np) else 50
    chop = np.where(atr_ma_1d > 0, 100 * np.log10(atr_sum / (np.log(14)/np.log(10) * atr_ma_1d)) / np.log(10), 50)
    chop = np.where(~np.isnan(chop), chop, 50)
    
    # Proper CHOP calculation
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop_denom = np.log(14) * atr_ma_1d
    chop = np.where(chop_denom != 0, 100 * np.log10(atr_sum / chop_denom), 50)
    chop = np.where(~np.isnan(chop), chop, 50)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align 1d indicators to 6h timeframe
    chop_6h = align_htf_to_ltf(prices, df_1d, chop)
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Warmup for TRIX and indicators
        # Skip if required data not available
        if (np.isnan(trix[i]) or 
            np.isnan(chop_6h[i]) or 
            np.isnan(ema50_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: TRIX turns negative OR chop indicates strong trend (avoid whipsaw)
            if trix[i] < 0 or chop_6h[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: TRIX turns positive OR chop indicates strong trend
            if trix[i] > 0 or chop_6h[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: TRIX positive + chop > 61.8 (range) + volume confirmation + price above EMA50
            if (trix[i] > 0 and 
                chop_6h[i] > 61.8 and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: TRIX negative + chop > 61.8 (range) + volume confirmation + price below EMA50
            elif (trix[i] < 0 and 
                  chop_6h[i] > 61.8 and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals