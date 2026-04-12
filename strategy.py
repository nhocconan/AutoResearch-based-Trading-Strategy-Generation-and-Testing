#!/usr/bin/env python3
"""
4h_1d_TRIX_Volume_Regime_v1
Hypothesis: On 4h timeframe, use TRIX momentum with daily volume confirmation and volatility regime filter.
TRIX > 0 indicates bullish momentum, TRIX < 0 indicates bearish momentum.
Volume confirmation ensures institutional participation.
Volatility regime filter avoids choppy markets.
Designed for low trade frequency (20-40/year) by requiring multiple confluence factors.
Works in bull/bear via TRIX momentum and volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_TRIX_Volume_Regime_v1"
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
    
    # === DAILY TRIX (15-period) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX: triple EMA of percentage price change
    # TRIX = EMA(EMA(EMA(ROC, 15), 15), 15)
    roc = np.diff(close_1d) / close_1d[:-1] * 100
    roc = np.concatenate([[0], roc])  # first value is 0
    
    # Triple EMA
    ema1 = np.zeros_like(roc)
    ema2 = np.zeros_like(roc)
    ema3 = np.zeros_like(roc)
    
    alpha = 2 / (15 + 1)
    for i in range(len(roc)):
        if i == 0:
            ema1[i] = roc[i]
            ema2[i] = roc[i]
            ema3[i] = roc[i]
        else:
            ema1[i] = alpha * roc[i] + (1 - alpha) * ema1[i-1]
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
    
    trix = ema3
    
    # === DAILY VOLUME AVERAGE (20-period) ===
    vol_20 = np.zeros_like(volume)
    vol_sum = 0.0
    vol_count = 0
    for i in range(len(volume)):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_20[i] = vol_sum / vol_count
        else:
            vol_20[i] = 0.0
    
    # === DAILY VOLATILITY REGIME (ATR-based) ===
    # ATR(14) for volatility measurement
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.nan
        elif i == 14:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volatility regime: low volatility when ATR < 20-period MA
    atr_ma = np.zeros_like(atr)
    atr_sum = 0.0
    atr_count = 0
    for i in range(len(atr)):
        if np.isnan(atr[i]):
            atr_sum = 0.0
            atr_count = 0
            atr_ma[i] = np.nan
        else:
            atr_sum += atr[i]
            atr_count += 1
            if i >= 20:
                atr_sum -= atr[i-20]
                atr_count -= 1
            if atr_count > 0:
                atr_ma[i] = atr_sum / atr_count
            else:
                atr_ma[i] = 0.0
    
    # Low volatility regime (trending) when ATR < MA
    vol_regime = atr < atr_ma
    
    # Align data to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    vol_20_aligned = align_htf_to_ltf(prices, df_1d, vol_20)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if indicators not available
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_20_aligned[i]) or 
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_20_aligned[i]
        
        # Only trade in low volatility (trending) regime
        in_trend_regime = vol_regime_aligned[i] > 0.5
        
        # TRIX thresholds
        trix_threshold = 0.1  # small threshold to avoid whipsaw
        
        # Entry conditions
        long_setup = (trix_aligned[i] > trix_threshold) and vol_confirm and in_trend_regime
        short_setup = (trix_aligned[i] < -trix_threshold) and vol_confirm and in_trend_regime
        
        # Exit conditions: TRIX crosses zero
        exit_long = trix_aligned[i] < 0
        exit_short = trix_aligned[i] > 0
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals