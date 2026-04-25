#!/usr/bin/env python3
"""
4h TRIX + Volume Spike + Choppiness Regime Filter
Hypothesis: TRIX (12,20,9) identifies momentum shifts; combined with volume spike (>2x 20-bar average) 
and choppiness regime (CHOP(14) > 61.8 = ranging market for mean reversion, CHOP < 38.2 = trending), 
we capture high-probability reversals in both bull and bear markets. Uses discrete sizing (0.25) to minimize fee churn.
Target: 20-40 trades/year on 4h.
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
    
    # Calculate TRIX on close: EMA(EMA(EMA(close,12),12),12) then ROC
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    if len(close) < 40:  # Need enough for triple EMA + ROC
        return np.zeros(n)
    
    ema1 = ema(close, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    trix_raw = np.full(n, np.nan)
    trix_raw[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (HHV(high,14) - LLV(low,14)))
    if len(high) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
        
        # Sum of ATR over 14 periods
        atr_sum = np.full(n, np.nan)
        for i in range(13, n):
            atr_sum[i] = np.sum(tr[i-13:i+1])
        
        # Highest high and lowest low over 14 periods
        hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
        
        # Chop = 100 * log10(atr_sum) / log10(14) / (hh - ll)
        chop = np.full(n, np.nan)
        mask = (hh - ll) > 0
        chop[mask] = 100 * np.log10(atr_sum[mask]) / np.log10(14) / (hh[mask] - ll[mask])
    else:
        chop = np.full(n, np.nan)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    for i in range(0, 19):
        vol_ma_20[i] = np.mean(volume[:i+1])
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_raw[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        trix = trix_raw[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        
        # TRIX signal: crossing zero line
        trix_cross_up = trix > 0 and (i == start_idx or trix_raw[i-1] <= 0)
        trix_cross_down = trix < 0 and (i == start_idx or trix_raw[i-1] >= 0)
        
        if position == 0:
            # Long: TRIX crosses up AND volume spike AND chop < 38.2 (trending) OR chop > 61.8 (mean reversion in ranging)
            # In trending markets (chop < 38.2): follow TRIX momentum
            # In ranging markets (chop > 61.8): mean reversion - long when TRIX turns up from negative
            long_condition = vol_spike and (
                (chop_val < 38.2 and trix_cross_up) or  # Trending: follow momentum
                (chop_val > 61.8 and trix > -0.1 and trix_raw[i-1] < -0.1)  # Ranging: bullish reversal from oversold
            )
            # Short: TRIX crosses down AND volume spike AND chop < 38.2 (trending) OR chop > 61.8 (mean reversion in ranging)
            short_condition = vol_spike and (
                (chop_val < 38.2 and trix_cross_down) or  # Trending: follow momentum
                (chop_val > 61.8 and trix < 0.1 and trix_raw[i-1] > 0.1)  # Ranging: bearish reversal from overbought
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: TRIX turns down OR chop increases significantly (trend weakening)
            if trix < 0 or chop_val > 55.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX turns up OR chop increases significantly (trend weakening)
            if trix > 0 or chop_val > 55.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0