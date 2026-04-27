#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_v1
Hypothesis: TRIX (15-period) crossover signals combined with volume spike confirmation and choppiness regime filter (CHOP > 61.8 for mean reversion). Enters long when TRIX crosses above zero with volume spike in choppy market, short when TRIX crosses below zero with volume spike in choppy market. Uses ATR-based stoploss for risk control. Designed for low trade frequency (target: 20-50/year) to minimize fee drag. Works in both bull and bear markets by using TRIX for momentum and chop regime to avoid trending whipsaws.
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
    
    # Calculate TRIX (15-period): EMA of EMA of EMA of close, then ROC
    def ema(series, span):
        return pd.Series(series).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema1 = ema(close, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    # Avoid division by zero
    ema3_safe = np.where(ema3 == 0, 1e-10, ema3)
    trix = (ema3 - np.roll(ema3, 1)) / ema3_safe * 100
    trix[0] = 0  # first value has no previous
    
    # Calculate ATR (14-period) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14-period): measures sideways vs trending market
    # CHOP = 100 * log10(sum(atr14) / (log10(highest_high - lowest_low) * 14)) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high - lowest_low
    # Avoid log of zero or negative
    range_14_safe = np.where(range_14 <= 0, 1e-10, range_14)
    chop = 100 * (np.log10(sum_atr_14) - np.log10(range_14_safe) - np.log10(14)) / (-np.log10(14))
    chop = np.where(range_14_safe == 1e-10, 50, chop)  # set to neutral when range invalid
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for TRIX (3*15=45), ATR (14), CHOP (14), volume avg (20)
    start_idx = max(50, 45, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        trix_val = trix[i]
        trix_prev = trix[i-1]
        atr_val = atr[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: TRIX crossover with volume spike in choppy market (CHOP > 61.8)
            # Long: TRIX crosses above zero AND volume spike AND choppy market
            # Short: TRIX crosses below zero AND volume spike AND choppy market
            trix_cross_up = trix_prev <= 0 and trix_val > 0
            trix_cross_down = trix_prev >= 0 and trix_val < 0
            choppy_market = chop_val > 61.8
            
            if trix_cross_up and vol_spike and choppy_market:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif trix_cross_down and vol_spike and choppy_market:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when TRIX crosses below zero (momentum loss) or ATR stoploss hit
            trix_cross_down = trix_prev >= 0 and trix_val < 0
            if trix_cross_down or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TRIX crosses above zero (momentum loss) or ATR stoploss hit
            trix_cross_up = trix_prev <= 0 and trix_val > 0
            if trix_cross_up or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0