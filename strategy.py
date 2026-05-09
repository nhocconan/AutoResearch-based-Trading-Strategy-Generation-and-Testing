#!/usr/bin/env python3
# 6h_Trix_VolumeSpike_Regime
# Hypothesis: Use TRIX (triple EMA) momentum with volume spike and Chop regime filter.
# Long when TRIX crosses above zero with volume > 2x average and Chop > 61.8 (ranging market).
# Short when TRIX crosses below zero with volume > 2x average and Chop > 61.8.
# Works in ranging markets by fading momentum extremes; avoids trending markets where TRIX whipsaws.
# Designed for 15-30 trades/year on 6h timeframe with strict entry conditions.

name = "6h_Trix_VolumeSpike_Regime"
timeframe = "6h"
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
    
    # === TRIX Calculation (15-period triple EMA) ===
    # EMA1
    ema1 = np.full_like(close, np.nan)
    if len(close) >= 15:
        ema1[14] = np.mean(close[0:15])
        for i in range(15, len(close)):
            ema1[i] = (close[i] * 2 + ema1[i-1] * 13) / 15
    
    # EMA2 of EMA1
    ema2 = np.full_like(close, np.nan)
    valid_ema1 = ~np.isnan(ema1)
    if np.sum(valid_ema1) >= 15:
        idx = np.where(valid_ema1)[0]
        start = idx[14]
        ema2[start] = np.mean(ema1[idx[0:15]])
        j = 15
        for i in range(start+1, len(close)):
            if valid_ema1[i]:
                ema2[i] = (ema1[i] * 2 + ema2[i-1] * 13) / 15
                j += 1
    
    # EMA3 of EMA2
    ema3 = np.full_like(close, np.nan)
    valid_ema2 = ~np.isnan(ema2)
    if np.sum(valid_ema2) >= 15:
        idx = np.where(valid_ema2)[0]
        start = idx[14]
        ema3[start] = np.mean(ema2[idx[0:15]])
        j = 15
        for i in range(start+1, len(close)):
            if valid_ema2[i]:
                ema3[i] = (ema2[i] * 2 + ema3[i-1] * 13) / 15
                j += 1
    
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix = np.full_like(close, np.nan)
    valid_ema3 = ~np.isnan(ema3)
    for i in range(1, len(close)):
        if valid_ema3[i] and valid_ema3[i-1] and ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # === Chop Index Calculation (14-period) ===
    # True Range
    tr = np.maximum(high - low, np.absolute(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]])))
    tr = np.maximum(tr, np.absolute(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]])))
    
    # ATR14
    atr = np.full_like(close, np.nan)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of absolute price changes
    abs_close_chg = np.absolute(np.diff(close, prepend=close[0]))
    sum_abs_chg = np.full_like(close, np.nan)
    if len(abs_close_chg) >= 14:
        sum_abs_chg[13] = np.sum(abs_close_chg[0:14])
        for i in range(14, len(abs_close_chg)):
            sum_abs_chg[i] = sum_abs_chg[i-1] - abs_close_chg[i-14] + abs_close_chg[i]
    
    # Chop = 100 * log10(sum(abs(close_chg))/(ATR*14)) / log10(14)
    chop = np.full_like(close, np.nan)
    valid = (~np.isnan(sum_abs_chg)) & (~np.isnan(atr)) & (atr != 0)
    for i in range(len(close)):
        if valid[i] and atr[i] * 14 > 0:
            chop[i] = 100 * np.log10(sum_abs_chg[i] / (atr[i] * 14)) / np.log10(14)
    
    # === Volume Spike Filter ===
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 15, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix[i-1]) or np.isnan(trix[i]) or \
           np.isnan(chop[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND volume spike AND Chop > 61.8 (ranging)
            if trix[i-1] <= 0 and trix[i] > 0 and volume_ratio[i] > 2.0 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND volume spike AND Chop > 61.8 (ranging)
            elif trix[i-1] >= 0 and trix[i] < 0 and volume_ratio[i] > 2.0 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR Chop < 38.2 (trending market)
            if trix[i] < 0 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR Chop < 38.2 (trending market)
            if trix[i] > 0 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals